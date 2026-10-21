# Beaver's Choice Paper Company: Multi-Agent System Reflection Report

**Deliverables**
- Implementation: `project_starter.py` (a single Python file)
- Workflow diagram: `agent_workflow.png`, rendered from `agent_workflow.dot`
- Evaluation output: `test_results.csv`
- Business advisor output: `advisor_report.txt`

**Framework:** smolagents 1.26, using `ToolCallingAgent` with managed agents.
**Model:** `gpt-4o-mini`, called through the OpenAI-compatible Vocareum endpoint.

---

## 1. System overview

The system automates the company's request-to-cash flow. It reads a free-text customer request and maps
each product to the catalog. It then checks stock and supplier lead times against the customer's deadline
and buys missing stock when that is both useful and affordable. Finally it quotes the order with a
volume-discount policy, records the sale, and writes a transparent reply to the customer.

It uses **five agents**, which is the maximum allowed:

| # | Agent | Role | Tools (and the starter helpers each one uses) |
|---|-------|------|------------------------------------------------|
| 1 | **Orchestrator** | Customer-facing coordinator. Extracts the request date and deadline, delegates to the specialists in order (inventory → quoting → sales) and writes the final reply. It has no database access. | `get_order_status` reads the per-request order draft and returns the next step plus a customer-safe summary |
| 2 | **Inventory agent** | Owns stock levels and supplier purchases. Maps customer wording to catalog items, checks feasibility against the deadline and decides on reorders. It records every `stock_orders` transaction. | `find_catalog_items` (internal catalog matcher)<br>`get_inventory_snapshot` → **`get_all_inventory`**<br>`check_availability` → **`get_stock_level`**, **`get_supplier_delivery_date`**<br>`place_supplier_orders` → **`get_cash_balance`**, **`create_transaction`**, **`get_supplier_delivery_date`** |
| 3 | **Quoting agent** | Owns pricing. Looks at similar historical quotes and prices the feasible lines with a deterministic discount policy. | `search_past_quotes` → **`search_quote_history`**<br>`calculate_quote` (pricing policy) |
| 4 | **Sales agent** | Owns sales transactions. Finalizes the order once and records every `sales` transaction. | `check_cash_balance` → **`get_cash_balance`**<br>`finalize_sale` → **`get_stock_level`**, **`generate_financial_report`** (integrity check on orders over $1,000), **`create_transaction`** |
| 5 | **Business advisor** | Runs once after all requests. Analyses the transactions and recommends operational changes. This is the "stand out" suggestion from the rubric. | `get_financial_report` → **`generate_financial_report`**<br>`get_transaction_summary` (SQL on `transactions`) |

All seven required helpers (`create_transaction`, `get_all_inventory`, `get_stock_level`,
`get_supplier_delivery_date`, `get_cash_balance`, `generate_financial_report`, `search_quote_history`)
are used inside at least one tool. `init_database` and `generate_sample_inventory` are used by the test
harness to seed the database.

### 1.1 Starter helper functions, reviewed

| Helper | What it does | How the system uses it |
|--------|--------------|------------------------|
| `generate_sample_inventory` | Picks 40% of the catalog (fixed seed) and gives each item a stock of 200–800 and a minimum level of 50–150. | Called indirectly through `init_database`. The `min_stock_level` values drive the reorder buffer. |
| `init_database` | Rebuilds every table, loads the historical quotes and requests, and seeds $50,000 of cash plus the initial stock purchases dated 2025-01-01. | Called once at the start of `run_test_scenarios`. |
| `create_transaction` | Inserts a `stock_orders` or `sales` row (units, total price, date) and returns its id. | The inventory agent records purchases with it and the sales agent records sales with it. |
| `get_all_inventory` | Returns net stock per item (purchases minus sales) up to a date, keeping only items with positive stock. | `get_inventory_snapshot` gives the inventory agent an overview of stock. |
| `get_stock_level` | Returns the net stock of a single item up to a date. | Used by `check_availability` for feasibility and by `finalize_sale` to re-check stock just before selling. |
| `get_supplier_delivery_date` | Supplier lead time grows with quantity: same day up to 10 units, 1 day up to 100, 4 days up to 1,000, 7 days above that. | Compared with the deadline to decide feasibility. Also reported as the expected arrival of purchases. |
| `get_cash_balance` | Sales revenue minus purchase cost up to a date. | Enforces the cash reserve before each supplier purchase, and lets the sales agent sanity-check large orders. |
| `generate_financial_report` | Cash, inventory value, total assets, stock per item and the top 5 sellers. | Integrity check before large sales, input for the business advisor, and per-request state in the test loop. |
| `search_quote_history` | Keyword search (all terms must match) over historical requests and quote explanations. | `search_past_quotes` calls it with one term at a time so it behaves like an OR search, and filters out corrupted rows. |

## 2. Workflow and design decisions

**Per-request flow** (see `agent_workflow.png`):

1. The orchestrator receives the request text with its date and extracts the deadline.
2. The **inventory agent**:
   - calls `find_catalog_items`;
   - calls `check_availability` with the sold lines (quantities as written, with their unit) and the
     list of products we do not sell;
   - if `next_step` requires it, calls `place_supplier_orders`.

   `check_availability` opens the **active order draft** for the request.
3. The orchestrator calls `get_order_status`, which says what to do next:
   - if at least one line is feasible, ask the **quoting agent** to price the draft, then the
     **sales agent** to finalize it;
   - if no line is feasible, reply right away with the reasons.
4. The orchestrator writes the reply **only** from the customer-safe summary returned by
   `get_order_status`.

**Key decisions, and why**

- **Responsibilities split by transaction type.** Only the inventory agent writes `stock_orders` and only
  the sales agent writes `sales`. Pricing has no database writes at all. This keeps responsibilities from
  overlapping and makes the audit trail obvious.
- **A shared order draft instead of passing data between LLMs.** The first prototypes passed item names and
  quantities between agents as prose. `gpt-4o-mini` often lost or rewrote them. For example, it forwarded
  "A4 glossy paper" instead of the catalog name "Glossy paper", turned "500 sheets" into "1 ream", and once
  invented an order id. Now each request is a session with a single active draft. `check_availability`
  creates it, and every later tool reads and updates it. LLMs decide *what* to do, while exact data never
  travels through them.
- **Deterministic business rules inside tools.** Feasibility, reorder quantities, pricing, discounts and
  rounding are Python code, so totals are reproducible and quotes always match the amount charged.
  `finalize_sale` recomputes the price with the same function as `calculate_quote`.
- **A status tool that acts as a state machine for the orchestrator.** During evaluation the orchestrator
  sometimes stopped after a partial inventory result, which left purchased stock unsold. `get_order_status`
  returns an explicit `next_step`. Together with `parallel_tool_calls=False`, this made delegation reliable:
  20 out of 20 requests followed the correct path in the final run.
- **Safeguards.**
  - Supplier purchases never take cash below a $5,000 reserve.
  - Sales are idempotent: an order can be finalized only once.
  - Stock is re-checked when the sale is made.
  - Large orders get a financial integrity check first.
  - Exceptions are logged internally, and the customer only sees a polite message.

**Business policy (stated assumptions)**
- Catalog `unit_price` is the customer list price. The supplier charges 60% of list price.
- Volume discounts apply to the total units ordered: 1,000+ units 5%, 5,000+ units 10%, 10,000+ units 15%.
  Totals above $10 are rounded down to the whole dollar in the customer's favor, as the historical quotes did.
- Reorder quantity is the shortfall plus enough extra to keep the item at its minimum stock level. If that
  buffer would miss the deadline, only the shortfall is ordered.
- 1 ream = 500 sheets.
- Size-only requests for sizes we do not stock, such as "A3 paper", are declined. A size combined with a
  paper type is mapped to that type, for example "A3 matte paper" becomes "Matte paper".
- Purchases and sales are dated on the request date. The customer's delivery date is the supplier's
  arrival date when a restock is needed, and the request date otherwise.

## 3. Evaluation results (`test_results.csv`)

All 20 requests in `quote_requests_sample.csv` were processed in date order, with no internal errors.

| Metric | Result | Rubric requirement |
|--------|--------|--------------------|
| Requests that changed the cash balance | **19 / 20** | ≥ 3 |
| Requests with at least one item sold | **19 / 20** | ≥ 3 |
| Requests fully declined, with reasons | **1** (#15) | at least one request not (fully) fulfilled |
| Requests partially fulfilled, with a reason for each declined item | **9** (#2, 3, 13, 14, 16, 17, 18, 19, 20) | — |
| Sales / supplier purchases recorded | 40 sales lines ($12,290.00) / 33 purchases ($8,037.33) | — |
| Cash 2025-04-01 → final | $45,059.70 → **$49,312.37** | — |
| Total assets (cash + inventory) | $50,000.00 → **$53,555.22** | — |

The declines all have real causes:

- **Products we do not carry:** balloons (#2), A3 paper (#3), tickets and finished posters (#20).
- **Deadlines the supplier's lead times cannot meet:**
  - #13: 500 sheets of A4 needed by 04-10, but the earliest delivery is 04-12;
  - #14 and #15: orders above 1,000 units need 7 days of lead time;
  - #16 to #19: requests arrive 1–3 days before the deadline.
- **#15** (10,000 A4, 5,000 colored and 500 reams of cardstock, due in 3 days) could not be served at
  all. The customer received the earliest possible delivery date for each item.

### Strengths
- **Transparent replies.** Every reply lists the confirmed items, quantities, line prices and delivery
  dates. It shows subtotal, volume discount (rate and reason), rounding credit and total, and gives a
  reason for **every** declined item, including the earliest possible delivery date when the problem is
  timing.
- **No internal leakage.** Replies are written from a customer-safe summary. Stock levels, costs, margins,
  cash, supplier details and transaction ids never reach the customer. Earlier versions leaked messages
  such as "143 sheets in stock" or "cash reserve issues", and moving to the summary fixed this.
- **Profitable and consistent.** Every sale is priced above wholesale cost even at the 15% tier. Quotes
  match the amounts charged exactly. Cash grew by $4,252.67 over the run.
- **Good use of inventory.** Items in stock ship the same day. Back-to-back supplier orders let the company
  accept large orders it did not have in stock, such as 255,000 sheets of copy paper in #3, as long as the
  deadline allowed it.
- **Robust orchestration.** Each specialist was called once, in the right order, for every request.

### Areas for improvement observed
- **Stock-outs after big orders.** When the min-level buffer would miss the deadline, only the shortfall is
  bought, so A4 and colored paper ended at zero stock. Later requests for them then hit 4–7 day lead
  times, and that alone explains most of the declines in #14–#19. The business advisor agent raised this
  independently in `advisor_report.txt`.
- **In-transit stock counts as available.** Purchases are dated on the order date, so later requests see
  stock that has not arrived yet.
- **Matching is heuristic.** Wording such as "cardstock in various colors" or "A3 glossy paper" maps to the
  closest generic item, which a real customer might not accept without confirmation.
- **Replies vary in wording.** The figures are always correct, but layout and phrasing differ between
  requests because the final text is still LLM-generated.

## 4. Suggestions for further improvement

1. **Proactive replenishment agent or policy.** After each sale, and daily, reorder any item that falls
   below its minimum level, separately from customer orders. Size reorders from recent demand (for example
   a reorder point equal to lead-time demand plus safety stock) and stay within the cash reserve. This
   targets the stock-outs that caused most declines. The advisor's recommendations could feed this policy
   automatically.
2. **Partial delivery and negotiation instead of a flat decline.** When only part of a quantity can arrive
   by the deadline, offer to ship what is in stock now and the rest at the earliest supplier date, or
   propose a substitute item. A customer agent built from each request's context (job, event, order size)
   could accept or counter these offers, and the system would learn which concessions convert.
3. **Arrival-dated inventory.** Record purchases on the supplier's arrival date and keep an "on order"
   view, so feasibility checks see real on-hand stock. Pair this with a confirmation step whenever catalog
   matching relies on a fuzzy or size-only match.
4. **Data-driven pricing.** Use the historical quotes, not just the volume tiers, to adapt discounts by
   order size and event type, while protecting a minimum margin.

## 5. Requirements compliance

| Rubric requirement | How and where it is met |
|--------------------|-------------------------|
| **Diagram:** all agents shown, max 5 | `agent_workflow.png` shows the 5 agents (orchestrator, inventory, quoting, sales, business advisor). |
| **Diagram:** distinct, non-overlapping responsibilities | Each agent cluster states its role. Inventory alone writes `stock_orders`, sales alone writes `sales`, quoting only prices (section 2). |
| **Diagram:** clear orchestration logic and data flow | Numbered edges show orchestrator → inventory (1) → quoting (2) → sales (3), the returned data, and the shared order draft. |
| **Diagram:** tools tied to agents, with purpose and helper functions | Every tool box sits inside its agent's cluster and lists its purpose and the starter helpers it calls. |
| **Diagram:** agent–tool interactions (input/output) | Agent–tool edges are labelled `input -> output`, e.g. `lines, dates, not-sold -> feasibility`. |
| **Implementation** matches the diagram | `build_agent_system()` creates exactly the agents and tool sets drawn in the diagram. |
| **Orchestrator** delegates tasks | The `orchestrator` `ToolCallingAgent` has the three operational agents as `managed_agents` and follows `get_order_status`. |
| **Distinct operational agents** for inventory, quoting and sales finalization | `inventory_agent`, `quoting_agent` and `sales_agent`, each with its own tools and instructions. |
| **Recommended framework** | smolagents (`ToolCallingAgent`, `OpenAIServerModel`, `@tool`). |
| **Tools follow framework conventions** | Every tool is a `@tool` function with type hints and a docstring with `Args:`, which smolagents turns into the tool schema. |
| **All 7 helper functions used in tool definitions** | Called directly inside `@tool` functions. See the tool column of the table in section 1 and section 1.1. |
| **Evaluated on the full `quote_requests_sample.csv`** | `run_test_scenarios()` processes all 20 requests and writes `test_results.csv` (20 rows). |
| **≥ 3 requests change the cash balance** | 19 of 20 (section 3). |
| **≥ 3 quote requests fulfilled** | 19 of 20 (section 3). |
| **Not all requests fulfilled, with reasons** | #15 fully declined, 9 partially declined. Every declined item has a reason in the reply (section 3). |
| **Report:** diagram explanation and design rationale | Sections 1 and 2. |
| **Report:** discussion of results and strengths | Section 3. |
| **Report:** ≥ 2 improvement suggestions | Section 4 (four suggestions). |
| **Customer outputs** contain all relevant information | Items, quantities, line prices, delivery dates, subtotal, discount, total and declined items. |
| **Customer outputs** justify key decisions | Discount rate and tier, rounding credit, a reason for every declined item, and the earliest possible delivery date. |
| **Customer outputs** reveal no sensitive data or PII | Replies are written only from the customer-safe summary in `get_order_status`. Exceptions are replaced by a polite message (`handle_customer_request`). |
| **Readable code** (naming, comments, docstrings, modularity) | snake_case functions, UPPER_CASE policy constants, docstrings on every function and tool, and sections for policy, internal helpers, the tools of each agent, progress display, agent setup and test harness. |
| **Stand-out suggestions** | Business advisor agent (`advisor_report.txt`), plus a terminal progress animation showing which agent is working on each step (`show_agent_progress`). |

## 6. How to run

```bash
cd starter
pip install -r requirements.txt   # Python 3.10+ (required by smolagents)
cp .env.sample .env               # then add UDACITY_OPENAI_API_KEY
python project_starter.py
```

This rebuilds `munder_difflin.db`, processes every sample request with a live progress animation, and
writes `test_results.csv` and `advisor_report.txt`. Set `AGENT_VERBOSITY=1` in `.env` to see the full
smolagents traces as well.
