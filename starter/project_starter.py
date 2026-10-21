import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import math
import re
from difflib import SequenceMatcher
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import OpenAIServerModel, ToolCallingAgent, tool

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

dotenv.load_dotenv()

OPENAI_API_BASE = "https://openai.vocareum.com/v1"
MODEL_ID = os.getenv("MODEL_ID", "gpt-4o-mini")


def build_model() -> OpenAIServerModel:
    """Create the OpenAI-compatible model shared by every agent in the system."""
    api_key = os.getenv("UDACITY_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("UDACITY_OPENAI_API_KEY is missing: add it to a .env file next to this script.")
    # One tool call per step keeps the delegation order (inventory -> quoting -> sales) strictly sequential.
    return OpenAIServerModel(model_id=MODEL_ID, api_base=OPENAI_API_BASE, api_key=api_key,
                             temperature=0, parallel_tool_calls=False)


########################
# Business policy
########################

# Catalog prices in `paper_supplies` are customer list prices. The supplier sells to us at a
# wholesale discount, which is where the company's margin comes from. Never shown to customers.
WHOLESALE_COST_RATIO = 0.60

# Supplier orders are refused if they would push cash below this safety reserve.
MIN_CASH_RESERVE = 5000.0

# Orders whose quoted total exceeds this value trigger a full financial integrity check before the sale.
LARGE_ORDER_THRESHOLD = 1000.0

# Volume discount tiers, applied on the total number of units in the order: (min_units, rate).
BULK_DISCOUNT_TIERS = [
    (10000, 0.15),
    (5000, 0.10),
    (1000, 0.05),
    (0, 0.0),
]

# Sheets per ream, used when customers order paper by the ream.
SHEETS_PER_REAM = 500

CATALOG = {item["item_name"]: item for item in paper_supplies}

# Shared order drafts, keyed by order id. Each customer request is handled as a session with one
# active draft: the inventory agent opens it, then supplier orders, quoting and the sale all work on
# that draft, so exact item names, quantities and ids never have to pass between LLM agents.
ORDER_DRAFTS: Dict[str, Dict] = {}
ACTIVE_ORDER: Dict[str, Union[str, None]] = {"order_id": None}

# Common customer wording that string similarity alone maps poorly: phrase -> catalog item.
CATALOG_SYNONYMS = {
    "printer paper": "Standard copy paper",
    "printing paper": "Standard copy paper",
    "copy paper": "Standard copy paper",
    "white paper": "Standard copy paper",
    "poster board": "Large poster paper (24x36 inches)",
    "napkin": "Paper napkins",
    "streamer": "Party streamers",
    "washi": "Decorative adhesive tape (washi tape)",
    "cardboard": "Cardstock",
    "construction paper": "Construction paper",
    "envelope": "Envelopes",
    "flyer": "Flyers",
}


########################
# Internal helpers (not exposed to the LLM)
########################

def _validate_iso_date(date_str: str) -> str:
    """Return the YYYY-MM-DD part of `date_str`, raising ValueError if it is not a valid ISO date."""
    date_part = str(date_str).strip().split("T")[0]
    datetime.strptime(date_part, "%Y-%m-%d")
    return date_part


def _normalize_order_lines(items: List[Dict]) -> List[Dict]:
    """
    Coerce LLM-provided order lines into [{'item_name': str, 'quantity': int}] with positive quantities,
    converting reams to sheets and merging repeated items into a single line.
    """
    quantities: Dict[str, int] = {}
    for raw in items:
        name = str(raw.get("item_name", "")).strip()
        quantity = int(float(raw.get("quantity", 0)))
        if str(raw.get("unit", "")).strip().lower() in {"ream", "reams"}:
            quantity *= SHEETS_PER_REAM
        if name and quantity > 0:
            quantities[name] = quantities.get(name, 0) + quantity
    return [{"item_name": name, "quantity": quantity} for name, quantity in quantities.items()]


def _min_stock_levels() -> Dict[str, int]:
    """Read the reorder threshold of every item tracked in the `inventory` reference table."""
    inventory_df = pd.read_sql("SELECT item_name, min_stock_level FROM inventory", db_engine)
    return dict(zip(inventory_df["item_name"], inventory_df["min_stock_level"].astype(int)))


def _active_draft() -> Union[Dict, None]:
    """Return the order draft of the customer request currently being handled, if any."""
    order_id = ACTIVE_ORDER["order_id"]
    return ORDER_DRAFTS.get(order_id) if order_id else None


def _rank_catalog_matches(description: str) -> List[tuple]:
    """
    Score every catalog item against a customer description and return [(score, item_name)],
    best first, keeping only plausible matches. Combines token overlap, string similarity and
    a boost for known customer synonyms.
    """
    query = description.lower()
    query_tokens = set(re.findall(r"[a-z0-9]+", query))
    # "A4 paper" is a size-specific item: never offer it for another paper size (A3, A5).
    other_size_requested = bool(query_tokens & {"a3", "a5"})
    synonym_hits = {target for phrase, target in CATALOG_SYNONYMS.items() if phrase in query}
    # Plain A4 office paper is its own catalog item rather than generic copy paper.
    if "a4" in query_tokens and "Standard copy paper" in synonym_hits:
        synonym_hits = (synonym_hits - {"Standard copy paper"}) | {"A4 paper"}

    scored = []
    for name in CATALOG:
        if other_size_requested and name == "A4 paper":
            continue
        name_lower = name.lower()
        name_tokens = set(re.findall(r"[a-z0-9]+", name_lower))
        overlap = len(query_tokens & name_tokens) / max(len(name_tokens), 1)
        ratio = SequenceMatcher(None, query, name_lower).ratio()
        score = 0.6 * overlap + 0.4 * ratio + (1.0 if name in synonym_hits else 0.0)
        scored.append((round(score, 3), name))
    scored.sort(reverse=True)
    return [(score, name) for score, name in scored if score >= 0.35]


def _feasible_lines(draft: Dict) -> List[Dict]:
    """Return the feasible lines of an order draft as [{'item_name', 'quantity'}]."""
    return [{"item_name": line["item_name"], "quantity": line["quantity"]}
            for line in draft["lines"] if line["feasible"]]


def _declined_items(draft: Dict) -> List[Dict]:
    """List what the customer asked for but will not receive, with a customer-friendly reason."""
    declined = [{"item": description, "reason": "not a product we carry"} for description in draft["not_sold"]]
    declined += [{"item": line["item_name"], "reason": line.get("reason", "cannot be supplied")}
                 for line in draft["lines"] if not line["feasible"]]
    return declined


def _bulk_discount_rate(total_units: int) -> float:
    """Return the volume discount rate that applies to an order of `total_units` units."""
    for min_units, rate in BULK_DISCOUNT_TIERS:
        if total_units >= min_units:
            return rate
    return 0.0


def _price_order(lines: List[Dict]) -> Dict:
    """
    Deterministically price an order from catalog list prices.

    Applies the volume discount tier, then rounds the total down to a whole dollar
    (customer-friendly rounding, as in the historical quotes) when the total is above $10.
    Each line receives its share of the final total so that sales transactions add up exactly.
    """
    total_units = sum(line["quantity"] for line in lines)
    discount_rate = _bulk_discount_rate(total_units)

    priced_lines = []
    for line in lines:
        unit_price = CATALOG[line["item_name"]]["unit_price"]
        list_total = round(unit_price * line["quantity"], 2)
        priced_lines.append({
            "item_name": line["item_name"],
            "quantity": line["quantity"],
            "unit_price": unit_price,
            "list_total": list_total,
        })

    subtotal = round(sum(line["list_total"] for line in priced_lines), 2)
    volume_discount = round(subtotal * discount_rate, 2)
    discounted = round(subtotal - volume_discount, 2)
    final_total = float(math.floor(discounted)) if discounted > 10 else discounted

    # Spread the final total across lines proportionally; the last line absorbs rounding cents.
    allocated = 0.0
    for index, line in enumerate(priced_lines):
        if index == len(priced_lines) - 1:
            line["line_total"] = round(final_total - allocated, 2)
        else:
            share = line["list_total"] / subtotal if subtotal else 0.0
            line["line_total"] = round(final_total * share, 2)
            allocated += line["line_total"]

    return {
        "lines": priced_lines,
        "total_units": total_units,
        "subtotal": subtotal,
        "discount_rate": discount_rate,
        "volume_discount": volume_discount,
        "rounding_credit": round(discounted - final_total, 2),
        "total": final_total,
    }


########################
# Tools for inventory agent
########################

@tool
def find_catalog_items(descriptions: list[str]) -> str:
    """
    Match free-text product descriptions from a customer request to exact catalog item names.

    Returns, for each description, up to three candidate catalog names ranked by similarity.
    Use the exact returned names in every other tool. A description with no good candidate
    is something the company does not sell.

    Args:
        descriptions: Product descriptions as written by the customer, e.g. ["A4 glossy paper", "balloons"].
    """
    results = {}
    for description in descriptions:
        results[description] = [
            {"item_name": name, "match_score": score} for score, name in _rank_catalog_matches(description)[:3]
        ]
    return json.dumps(results, indent=2)


@tool
def get_inventory_snapshot(as_of_date: str) -> str:
    """
    List every item currently in stock with its quantity, as of a date.

    Args:
        as_of_date: Date in YYYY-MM-DD format.
    """
    try:
        as_of_date = _validate_iso_date(as_of_date)
    except ValueError:
        return "ERROR: as_of_date must be in YYYY-MM-DD format."
    inventory = get_all_inventory(as_of_date)
    return json.dumps({name: int(stock) for name, stock in inventory.items()}, indent=2)


@tool
def check_availability(items: list[dict], request_date: str, deadline: str, not_sold: list[str]) -> str:
    """
    Open an order draft and check whether each line can be delivered by the customer's deadline.

    For each line it reports current stock, any shortfall, the supplier quantity to reorder
    (shortfall plus a buffer that keeps stock above the minimum level), the supplier delivery
    date and whether the line is feasible. The result becomes the active order of the current
    customer request, which supplier orders, quoting and the sale then work on.

    Args:
        items: Order lines, each an object {"item_name": exact catalog name, "quantity": number as the customer wrote it, "unit": "sheets", "units" or "reams"}.
        request_date: Date the customer placed the request, YYYY-MM-DD.
        deadline: Delivery date requested by the customer, YYYY-MM-DD.
        not_sold: The customer's descriptions of requested products we do not sell (empty list if none).
    """
    try:
        request_date = _validate_iso_date(request_date)
        deadline = _validate_iso_date(deadline)
    except ValueError:
        return "ERROR: request_date and deadline must be in YYYY-MM-DD format."

    min_levels = _min_stock_levels()
    report = []
    for line in _normalize_order_lines(items):
        name, quantity = line["item_name"], line["quantity"]
        if name not in CATALOG:
            report.append({"item_name": name, "quantity": quantity, "feasible": False,
                           "reason": "not a catalog item - call find_catalog_items for the exact name"})
            continue

        stock = int(get_stock_level(name, request_date)["current_stock"].iloc[0])
        shortfall = max(0, quantity - stock)
        min_level = min_levels.get(name, 0)
        # Reorder enough to cover the order and leave at least the minimum stock level afterwards.
        reorder_quantity = max(0, quantity + min_level - stock)

        if reorder_quantity > 0:
            supplier_date = get_supplier_delivery_date(request_date, reorder_quantity)
            # If the buffered order would arrive too late, fall back to ordering only the shortfall.
            if shortfall > 0 and supplier_date > deadline:
                reorder_quantity = shortfall
                supplier_date = get_supplier_delivery_date(request_date, reorder_quantity)
        else:
            supplier_date = None

        delivery_date = supplier_date if shortfall > 0 else request_date
        feasible = delivery_date <= deadline
        entry = {
            "item_name": name,
            "quantity": quantity,
            "current_stock": stock,
            "shortfall": shortfall,
            "reorder_quantity": reorder_quantity,
            "supplier_delivery_date": supplier_date,
            "customer_delivery_date": delivery_date,
            "feasible": feasible,
        }
        if not feasible:
            entry["reason"] = (f"this quantity cannot be delivered by the requested date "
                               f"(earliest possible delivery: {supplier_date})")
        report.append(entry)

    # Drop "not sold" descriptions that actually match an item of this order (LLM double counting).
    ordered_items = {line["item_name"] for line in report}
    not_sold = [str(description) for description in not_sold
                if not any(name in ordered_items for _, name in _rank_catalog_matches(str(description))[:1])]

    order_id = f"ORD-{len(ORDER_DRAFTS) + 1:04d}"
    ORDER_DRAFTS[order_id] = {
        "request_date": request_date,
        "deadline": deadline,
        "lines": report,
        "not_sold": not_sold,
        "supplier_orders_placed": False,
        "status": "assessed",
    }
    ACTIVE_ORDER["order_id"] = order_id
    needs_restock = any(line["feasible"] and line.get("reorder_quantity", 0) > 0 for line in report)
    next_step = ("REQUIRED: call place_supplier_orders before answering."
                 if needs_restock else "No supplier order needed.")
    return json.dumps({"order_id": order_id, "lines": report, "not_sold": not_sold, "next_step": next_step},
                      indent=2)


@tool
def place_supplier_orders() -> str:
    """
    Buy from the supplier the reorder quantities of every feasible line of the active order.

    Records one 'stock_orders' transaction per item on the request date. Each purchase is checked
    against the company's cash reserve first; a refused purchase makes its line infeasible
    when the order cannot be served from current stock.
    """
    draft = _active_draft()
    if draft is None:
        return "ERROR: no active order - call check_availability first."
    order_id = ACTIVE_ORDER["order_id"]
    if draft["supplier_orders_placed"]:
        return "Supplier orders were already placed for this order; nothing to do."

    order_date = draft["request_date"]
    results = []
    for line in draft["lines"]:
        if not line["feasible"] or line.get("reorder_quantity", 0) <= 0:
            continue
        name, quantity = line["item_name"], line["reorder_quantity"]
        cost = round(quantity * CATALOG[name]["unit_price"] * WHOLESALE_COST_RATIO, 2)
        if get_cash_balance(order_date) - cost < MIN_CASH_RESERVE:
            if line["shortfall"] > 0:
                line["feasible"] = False
                line["reason"] = "insufficient supply available for this quantity"
            results.append({"item_name": name, "quantity": quantity, "status": "refused",
                            "reason": "purchase would breach the cash reserve"})
            continue
        transaction_id = create_transaction(name, "stock_orders", quantity, cost, order_date)
        results.append({
            "item_name": name,
            "quantity": quantity,
            "status": "ordered",
            "transaction_id": transaction_id,
            "expected_arrival": get_supplier_delivery_date(order_date, quantity),
        })

    draft["supplier_orders_placed"] = True
    feasible = [line["item_name"] for line in draft["lines"] if line["feasible"]]
    return json.dumps({"order_id": order_id, "purchases": results, "feasible_items": feasible}, indent=2)


########################
# Tools for quoting agent
########################

@tool
def search_past_quotes(search_term: str) -> str:
    """
    Search historical quotes whose request or explanation mentions a term (e.g. "cardstock", "ceremony").

    Use single words or short phrases; call several times for different terms.

    Args:
        search_term: Keyword to look for in past requests and quote explanations.
    """
    matches = search_quote_history([search_term], limit=3)
    summary = [
        {
            "total_amount": match["total_amount"],
            "order_size": match["order_size"],
            "event_type": match["event_type"],
            "quote_explanation": match["quote_explanation"],
        }
        for match in matches
        if match["quote_explanation"] != "Error parsing response."
    ]
    return json.dumps(summary, indent=2) if summary else "No past quotes found for this term."


@tool
def calculate_quote() -> str:
    """
    Price the feasible lines of the active order from catalog list prices, applying the volume discount policy.

    Discount tiers on total units ordered: 1,000+ units 5%, 5,000+ units 10%, 10,000+ units 15%.
    Totals above $10 are rounded down to a whole dollar in the customer's favor.
    """
    draft = _active_draft()
    if draft is None:
        return "ERROR: the inventory check has not been done for this request yet."
    order_id = ACTIVE_ORDER["order_id"]
    lines = _feasible_lines(draft)
    if not lines:
        return "This order has no feasible lines to price."
    quote = _price_order(lines)
    draft["quote"] = quote
    return json.dumps({"order_id": order_id, **quote}, indent=2)


########################
# Tools for sales agent
########################

@tool
def finalize_sale() -> str:
    """
    Complete the sale of the active order: verify stock, record one 'sales' transaction per line, confirm.

    Prices are recomputed with the same policy as calculate_quote, so the charged total always
    matches the quote. Lines without enough stock on the sale date are not sold. Large orders
    get a financial integrity check (cash and inventory reconciliation) before being recorded.
    An order can only be finalized once.
    """
    draft = _active_draft()
    if draft is None:
        return "ERROR: the inventory check has not been done for this request yet."
    order_id = ACTIVE_ORDER["order_id"]
    if draft["status"] == "completed":
        return json.dumps({"status": "already_completed", **draft["sale"]}, indent=2)

    sale_date = draft["request_date"]
    sellable, rejected = [], []
    for line in _feasible_lines(draft):
        stock = int(get_stock_level(line["item_name"], sale_date)["current_stock"].iloc[0])
        if stock < line["quantity"]:
            rejected.append({**line, "reason": "insufficient stock on the sale date"})
        else:
            sellable.append(line)

    declined = _declined_items(draft) + [
        {"item": line["item_name"], "reason": line["reason"]} for line in rejected
    ]
    if not sellable:
        draft["status"] = "not_completed"
        return json.dumps({"status": "not_completed", "declined_items": declined}, indent=2)

    quote = _price_order(sellable)
    if quote["total"] > LARGE_ORDER_THRESHOLD:
        report = generate_financial_report(sale_date)
        if report["cash_balance"] < 0 or report["inventory_value"] < 0:
            return json.dumps({"status": "on_hold", "reason": "financial integrity check failed"}, indent=2)

    for line in quote["lines"]:
        line["transaction_id"] = create_transaction(
            line["item_name"], "sales", line["quantity"], line["line_total"], sale_date
        )

    delivery_dates = {line["item_name"]: line["customer_delivery_date"]
                      for line in draft["lines"] if line["feasible"]}
    draft["status"] = "completed"
    draft["sale"] = {
        "order_id": order_id,
        "sale_date": sale_date,
        "lines": [{**{k: line[k] for k in ("item_name", "quantity", "line_total", "transaction_id")},
                   "delivery_date": delivery_dates[line["item_name"]]}
                  for line in quote["lines"]],
        "subtotal": quote["subtotal"],
        "discount_rate": quote["discount_rate"],
        "volume_discount": quote["volume_discount"],
        "rounding_credit": quote["rounding_credit"],
        "total_charged": quote["total"],
        "declined_items": declined,
    }
    return json.dumps({"status": "completed", **draft["sale"]}, indent=2)

@tool
def check_cash_balance(as_of_date: str) -> str:
    """
    Return the company's cash balance as of a date (internal use only, never share with customers).

    Args:
        as_of_date: Date in YYYY-MM-DD format.
    """
    try:
        as_of_date = _validate_iso_date(as_of_date)
    except ValueError:
        return "ERROR: as_of_date must be in YYYY-MM-DD format."
    return f"Cash balance as of {as_of_date}: ${get_cash_balance(as_of_date):,.2f}"


########################
# Tools for business advisor agent
########################

@tool
def get_financial_report(as_of_date: str) -> str:
    """
    Produce the company financial report: cash, inventory value, total assets, stock per item and top sellers.

    Args:
        as_of_date: Date in YYYY-MM-DD format.
    """
    try:
        as_of_date = _validate_iso_date(as_of_date)
    except ValueError:
        return "ERROR: as_of_date must be in YYYY-MM-DD format."
    report = generate_financial_report(as_of_date)
    min_levels = _min_stock_levels()
    for entry in report["inventory_summary"]:
        entry["stock"] = int(entry["stock"])
        entry["min_stock_level"] = min_levels.get(entry["item_name"])
    return json.dumps(report, indent=2, default=str)


@tool
def get_transaction_summary(as_of_date: str) -> str:
    """
    Summarize all sales and supplier purchases after the opening day, per item, up to a date.

    Args:
        as_of_date: Date in YYYY-MM-DD format.
    """
    try:
        as_of_date = _validate_iso_date(as_of_date)
    except ValueError:
        return "ERROR: as_of_date must be in YYYY-MM-DD format."
    query = """
        SELECT item_name, transaction_type, COUNT(*) AS transactions,
               SUM(units) AS units, ROUND(SUM(price), 2) AS amount
        FROM transactions
        WHERE item_name IS NOT NULL
          AND transaction_date > '2025-01-01T00:00:00'
          AND transaction_date <= :as_of_date
        GROUP BY item_name, transaction_type
        ORDER BY transaction_type, amount DESC
    """
    summary = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})
    return summary.to_string(index=False) if not summary.empty else "No transactions recorded yet."


########################
# Tools for orchestrator agent
########################

@tool
def get_order_status() -> str:
    """
    Report the progress of the current customer request and the next action to take.

    Also returns a customer-safe summary (items, quantities, delivery dates, prices, declined items
    with reasons) to base the reply on. It contains no internal data such as stock levels or costs.
    """
    draft = _active_draft()
    if draft is None:
        return json.dumps({"next_step": "Call inventory_agent to assess the request."})

    feasible = [line for line in draft["lines"] if line["feasible"]]
    if not feasible:
        next_step = "Nothing can be supplied: reply to the customer now, explaining every declined item."
    elif "quote" not in draft:
        next_step = "Call quoting_agent to quote the current order."
    elif draft["status"] == "assessed":
        next_step = (f"Call sales_agent to finalize the current order (request date {draft['request_date']}, "
                     f"quoted total ${draft['quote']['total']:,.2f}).")
    else:
        next_step = "Order processed: reply to the customer now using customer_summary."

    summary = {"requested_delivery_by": draft["deadline"]}
    if draft["status"] == "completed":
        sale = draft["sale"]
        summary.update({
            "items_confirmed": [{k: line[k] for k in ("item_name", "quantity", "line_total", "delivery_date")}
                                for line in sale["lines"]],
            "subtotal": sale["subtotal"],
            "volume_discount_rate": sale["discount_rate"],
            "volume_discount": sale["volume_discount"],
            "rounding_credit": sale["rounding_credit"],
            "total_charged": sale["total_charged"],
            "declined_items": sale["declined_items"],
        })
    else:
        summary.update({
            "items_feasible": [{"item_name": line["item_name"], "quantity": line["quantity"],
                                "delivery_date": line["customer_delivery_date"]} for line in feasible],
            "declined_items": _declined_items(draft),
        })
    return json.dumps({"status": draft["status"], "next_step": next_step, "customer_summary": summary}, indent=2)


########################
# Terminal progress animation
########################

# smolagents log level: 0 shows only errors so the progress animation stays readable; 1 shows full agent traces.
AGENT_VERBOSITY = int(os.getenv("AGENT_VERBOSITY", "0"))

AGENT_LABELS = {
    "orchestrator": "Orchestrator",
    "inventory_agent": "Inventory",
    "quoting_agent": "Quoting",
    "sales_agent": "Sales",
    "business_advisor_agent": "Advisor",
}

# Completed tool calls in plain words. smolagents reports a step once it has finished, so each line
# describes something that has just happened.
TOOL_ACTIVITIES = {
    "inventory_agent": "received the inventory assessment",
    "quoting_agent": "received the quote",
    "sales_agent": "received the sale confirmation",
    "get_order_status": "checked the order status",
    "find_catalog_items": "matched products to the catalog",
    "get_inventory_snapshot": "reviewed current inventory",
    "check_availability": "checked stock and delivery deadline",
    "place_supplier_orders": "ordered missing stock from the supplier",
    "search_past_quotes": "looked up similar past quotes",
    "calculate_quote": "applied prices and volume discount",
    "check_cash_balance": "ran a pre-sale sanity check",
    "finalize_sale": "recorded the sale",
    "get_financial_report": "read the financial report",
    "get_transaction_summary": "reviewed all transactions",
    "final_answer": "finished",
}


def _animate_line(text: str, delay: float = 0.01) -> None:
    """Print a line with a typewriter effect when running in a terminal, instantly otherwise."""
    if not os.isatty(1):
        print(text)
        return
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def show_agent_progress(memory_step, agent=None) -> None:
    """smolagents step callback: show which agent did what, as each step completes."""
    tool_calls = getattr(memory_step, "tool_calls", None) or []
    label = AGENT_LABELS.get(getattr(agent, "name", ""), "Agent")
    indent = "  " if getattr(agent, "name", "") == "orchestrator" else "      "
    for tool_call in tool_calls:
        activity = TOOL_ACTIVITIES.get(tool_call.name, tool_call.name)
        _animate_line(f"{indent}[{label}] {activity}")


# Set up your agents and create an orchestration agent that will manage them.

INVENTORY_AGENT_INSTRUCTIONS = """
You are the inventory manager of a paper supply company. You own stock levels and supplier purchases.
Given a customer request, its request date and its delivery deadline:
1. Call find_catalog_items with every product the customer mentions and pick the exact catalog name
   for each, normally the top candidate. Size qualifiers (A3, A5, 8.5x11) are not catalog items: the
   paper type matters (e.g. "A3 matte paper" -> "Matte paper"). A size with no paper type (e.g. plain
   "A3 paper") is NOT SOLD, since we only stock A4 as a size. If no candidate genuinely is the same
   kind of product (e.g. balloons, tickets, finished posters), the item is NOT SOLD. Never substitute a
   different product (tickets are not sticky notes).
2. Call check_availability ONCE with all sold lines, and the customer's descriptions of NOT SOLD
   products in not_sold. Pass the quantity exactly as the customer wrote it with its unit
   ("sheets", "units" or "reams"); never convert quantities yourself.
3. Follow its next_step: if it says REQUIRED, call place_supplier_orders.
Final answer: each feasible item (customer description, catalog name, quantity, delivery date) and each
unavailable item with the customer's own description and a customer-friendly reason (not sold by us /
cannot be delivered by the deadline / insufficient supply). Never mention cash or stock levels.
"""

QUOTING_AGENT_INSTRUCTIONS = """
You are the quoting specialist of a paper supply company. The inventory check for the current customer
request has already been done; you price its feasible items.
1. Call search_past_quotes for one or two relevant terms (main product, event type) to see how similar
   orders were priced historically.
2. Call calculate_quote once. Its numbers are final: never change prices or totals.
Final answer: the per-line prices, subtotal, volume discount (rate and amount), the small rounding
credit if any (totals are rounded down to a whole dollar in the customer's favor), final total, and a short
customer-friendly explanation of the pricing (why the discount applies, comparison with similar past
orders if useful). Never mention costs, margins or supplier prices.
"""

SALES_AGENT_INSTRUCTIONS = f"""
You are the sales clerk who completes transactions for a paper supply company. The current customer
request has been checked and quoted; you receive its request date and quoted total.
1. If the quoted total is above ${LARGE_ORDER_THRESHOLD:,.0f}, call check_cash_balance on the request
   date first as a sanity check.
2. Call finalize_sale exactly once.
Final answer: the sale status, charged total, lines sold with delivery dates, and every declined item
with its reason, exactly as returned by finalize_sale.
"""

ADVISOR_AGENT_INSTRUCTIONS = """
You are a business advisor for a paper supply company. Use get_financial_report and
get_transaction_summary to analyze performance, then give 3-5 concrete, prioritized recommendations
to improve revenue and operational efficiency (stock levels, reorder policy, pricing, product range).
Base every recommendation on numbers from the tools.
"""

ORCHESTRATOR_INSTRUCTIONS = """
You are the customer service orchestrator of Beaver's Choice Paper Company. You never touch the
database yourself: you coordinate three specialists, each called at most once per request, in order.
1. Extract the request date (given at the end) and the delivery deadline (from the text; if missing,
   use 14 days after the request date). Dates are YYYY-MM-DD.
2. Call inventory_agent with the full list of products and quantities exactly as the customer wrote
   them, plus the request date and the deadline.
3. Then call get_order_status and do exactly what its next_step says. Call get_order_status again
   after every specialist, until it tells you to reply. If even one item is feasible, the order MUST be
   quoted (quoting_agent) and finalized (sales_agent) for those items, even if other items are declined.
Final answer = a reply addressed to the customer, written ONLY from get_order_status customer_summary:
- the confirmed items with quantities, line prices and delivery dates,
- subtotal, volume discount (rate and why: bulk order tiers 1,000/5,000/10,000+ units), rounding
  credit if any, and the total charged,
- EVERY declined item with its reason. Never drop one.
Never reveal internal information: stock levels, costs, margins, cash, supplier details, order or
transaction ids, tool names or error messages. Sign as "Beaver's Choice Paper Company - Customer Service".
"""


def build_agent_system(model: OpenAIServerModel) -> Dict[str, ToolCallingAgent]:
    """
    Create the five agents: the orchestrator with its three managed operational agents,
    plus the business advisor that reviews the company's results after the run.
    """
    inventory_agent = ToolCallingAgent(
        tools=[find_catalog_items, get_inventory_snapshot, check_availability, place_supplier_orders],
        model=model,
        name="inventory_agent",
        description=(
            "Inventory manager. Maps customer product descriptions to catalog items, checks stock and "
            "delivery feasibility against a deadline, and buys missing stock from the supplier. "
            "Give it the product descriptions with quantities, the request date and the deadline."
        ),
        instructions=INVENTORY_AGENT_INSTRUCTIONS,
        max_steps=8,
        verbosity_level=AGENT_VERBOSITY,
        step_callbacks=[show_agent_progress],
    )
    quoting_agent = ToolCallingAgent(
        tools=[search_past_quotes, calculate_quote],
        model=model,
        name="quoting_agent",
        description=(
            "Quoting specialist. Prices order lines (exact catalog names and quantities) with the "
            "volume discount policy and explains the price using past quotes."
        ),
        instructions=QUOTING_AGENT_INSTRUCTIONS,
        max_steps=6,
        verbosity_level=AGENT_VERBOSITY,
        step_callbacks=[show_agent_progress],
    )
    sales_agent = ToolCallingAgent(
        tools=[check_cash_balance, finalize_sale],
        model=model,
        name="sales_agent",
        description=(
            "Sales clerk. Records the sale of approved order lines (exact catalog names and quantities) "
            "on the request date and returns the confirmation."
        ),
        instructions=SALES_AGENT_INSTRUCTIONS,
        max_steps=4,
        verbosity_level=AGENT_VERBOSITY,
        step_callbacks=[show_agent_progress],
    )
    orchestrator = ToolCallingAgent(
        tools=[get_order_status],
        model=model,
        managed_agents=[inventory_agent, quoting_agent, sales_agent],
        name="orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        max_steps=8,
        verbosity_level=AGENT_VERBOSITY,
        step_callbacks=[show_agent_progress],
    )
    advisor_agent = ToolCallingAgent(
        tools=[get_financial_report, get_transaction_summary],
        model=model,
        name="business_advisor_agent",
        instructions=ADVISOR_AGENT_INSTRUCTIONS,
        max_steps=5,
        verbosity_level=AGENT_VERBOSITY,
        step_callbacks=[show_agent_progress],
    )
    return {"orchestrator": orchestrator, "advisor": advisor_agent}


def handle_customer_request(orchestrator: ToolCallingAgent, request_with_date: str) -> str:
    """Run one customer request through the orchestrator, hiding internal failures from the customer."""
    ACTIVE_ORDER["order_id"] = None  # each request starts a new order session
    try:
        return str(orchestrator.run(request_with_date))
    except Exception as error:
        print(f"INTERNAL ERROR while handling request: {error}")
        return ("We're sorry, we could not process your request right now. "
                "Our team will contact you shortly to complete it.")


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    agents = build_agent_system(build_model())
    orchestrator = agents["orchestrator"]

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = handle_customer_request(orchestrator, request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)

    # Business advisor review of the whole run
    print("\n===== BUSINESS ADVISOR RECOMMENDATIONS =====")
    advice = str(agents["advisor"].run(
        f"Review the company's performance from 2025-01-01 to {final_date} and recommend improvements."
    ))
    print(advice)
    with open("advisor_report.txt", "w") as advisor_file:
        advisor_file.write(advice)

    return results


if __name__ == "__main__":
    # The starter data files are read with relative paths, so run from this script's folder.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    results = run_test_scenarios()
