import json

from sqlalchemy.orm import Session

from app.models import SemanticConcept


# ---------------------------------------------------------------------------
# Family concepts (Part E). Generic types a data steward recognises. They are
# ranked ONLY when no specific concept wins (two-pass analysis). Every family
# carries rule_family: the downstream validity-rule family the rule stage can
# consume later (described in the report as a cross-stage follow-up).
# ---------------------------------------------------------------------------
FAMILIES = [
    {
        "concept_name": "Identifier",
        "category": "Family",
        "description": "Generic identifier: a code or number that names or refers to an entity. Uniqueness is expected only for an approved primary key, not for every identifier (foreign keys legitimately repeat).",
        "aliases": ["identifier", "id", "key", "reference number"],
        "expected_data_types": ["int", "string"],
        "profile_expectations": {
            "identifier_like": True,
            "non_null": True,
            "is_family": True,
            "family": "identifier",
            "rule_family": "pattern (shape/regex); uniqueness only if approved PK; referential integrity if FK",
            "role": "identifier",
            "head_tokens": ["identifier", "code", "number", "reference", "id"],
            "entities": [],
        },
    },
    {
        "concept_name": "Categorical",
        "category": "Family",
        "description": "Generic categorical attribute with a limited set of allowed values observed in the data.",
        "aliases": ["categorical", "category value", "classification value"],
        "expected_data_types": ["string", "int"],
        "profile_expectations": {
            "categorical": True,
            "non_null": True,
            "is_family": True,
            "family": "categorical",
            "rule_family": "allowed values (observed, user-approved); case/whitespace variants",
            "role": "categorical",
            "head_tokens": [],
            "entities": [],
        },
    },
    {
        "concept_name": "Money Amount",
        "category": "Family",
        "description": "Generic monetary amount: price, cost, fee, revenue or any value expressed in a currency.",
        "aliases": ["money", "monetary amount", "currency value", "amount"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "is_family": True,
            "family": "money",
            "rule_family": "numeric; non-negative (user confirms: refunds); decimals <=2; outliers as anomaly",
            "role": "measure",
            "head_tokens": ["amount", "price", "cost", "fee", "revenue", "sales", "salary", "total", "value"],
            "entities": [],
        },
    },
    {
        "concept_name": "Count",
        "category": "Family",
        "description": "Generic non-negative integer count of items, events or occurrences.",
        "aliases": ["count", "number of", "quantity"],
        "expected_data_types": ["int"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "is_family": True,
            "family": "count",
            "rule_family": "integer; non-negative; outliers as anomaly",
            "role": "count",
            "head_tokens": ["count", "quantity", "number"],
            "entities": [],
        },
    },
    {
        "concept_name": "Boolean Flag",
        "category": "Family",
        "description": "Generic boolean flag with two allowed states such as 0/1, yes/no or true/false.",
        "aliases": ["boolean flag", "flag", "boolean", "indicator"],
        "expected_data_types": ["bool", "int", "string"],
        "profile_expectations": {
            "boolean_like": True,
            "is_family": True,
            "family": "boolean_flag",
            "rule_family": "allowed values {0,1}/{yes,no}/{true,false}",
            "role": "flag",
            "head_tokens": [],
            "entities": [],
            "value_regex": r"^(0|1|[Tt]rue|[Ff]alse|[Yy]es|[Nn]o|[Yy]|[Nn])$",
        },
    },
    {
        "concept_name": "Free Text",
        "category": "Family",
        "description": "Generic free-text content that is not validated by a pattern, such as descriptions, comments or reviews.",
        "aliases": ["free text", "text", "notes", "comment"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "is_family": True,
            "family": "free_text",
            "rule_family": "completeness; max length; not validated by pattern",
            "role": "text",
            "head_tokens": ["text", "description", "comment", "notes", "review"],
            "entities": [],
        },
    },
    {
        "concept_name": "Person Name",
        "category": "Family",
        "description": "Generic personal name of a customer, contact, passenger or employee.",
        "aliases": ["person name", "full name", "customer name", "first name", "last name"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "is_family": True,
            "family": "person_name",
            "rule_family": "pattern: letters/spaces/punctuation; no digits; completeness",
            "role": "text",
            "head_tokens": ["name"],
            "entities": [],
        },
    },
    {
        "concept_name": "Organization Name",
        "category": "Family",
        "description": "Generic name of a company, supplier, shipper or any organization.",
        "aliases": ["organization name", "company name", "business name"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "is_family": True,
            "family": "organization_name",
            "rule_family": "free text; completeness; duplicates (consistency)",
            "role": "text",
            "head_tokens": ["name"],
            "entities": [],
        },
    },
    {
        "concept_name": "Product Name",
        "category": "Family",
        "description": "Generic product or item name.",
        "aliases": ["product name", "item name"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "is_family": True,
            "family": "product_name",
            "rule_family": "free text; completeness",
            "role": "text",
            "head_tokens": ["name"],
            "entities": [],
        },
    },
    {
        "concept_name": "City",
        "category": "Family",
        "description": "Generic city or locality name.",
        "aliases": ["city", "locality", "town"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "is_family": True,
            "family": "city",
            "rule_family": "reference list + consistency city->state->country",
            "role": "categorical",
            "head_tokens": [],
            "entities": [],
        },
    },
    {
        "concept_name": "Date",
        "category": "Family",
        "description": "Generic calendar date of an event, transaction or record.",
        "aliases": ["date", "calendar date"],
        "expected_data_types": ["date", "datetime", "string"],
        "profile_expectations": {
            "temporal": True,
            "is_family": True,
            "family": "date",
            "rule_family": "detected date format; parse rate; future/past plausibility; ordering vs other dates",
            "role": "date",
            "head_tokens": ["date", "day"],
            "entities": [],
        },
    },
    {
        "concept_name": "Timestamp",
        "category": "Family",
        "description": "Generic date-and-time moment when something happened.",
        "aliases": ["timestamp", "datetime"],
        "expected_data_types": ["datetime", "string"],
        "profile_expectations": {
            "temporal": True,
            "is_family": True,
            "family": "timestamp",
            "rule_family": "detected datetime format; parse rate; freshness candidate",
            "role": "time",
            "head_tokens": ["timestamp", "time", "at"],
            "entities": [],
        },
    },
    {
        "concept_name": "Measure",
        "category": "Family",
        "description": "Generic continuous numeric measurement such as temperature, area or distance.",
        "aliases": ["measure", "measurement"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "is_family": True,
            "family": "measure",
            "rule_family": "numeric; non-negative; unit consistency; outliers as anomaly",
            "role": "measure",
            "head_tokens": ["area", "distance", "temperature", "weight", "length", "width", "height", "size"],
            "entities": [],
        },
    },
    {
        "concept_name": "Postal Code",
        "category": "Family",
        "description": "Generic postal or ZIP code. Must be read as text to keep leading zeros.",
        "aliases": ["postal code", "zip code", "postcode"],
        "expected_data_types": ["string", "int"],
        "profile_expectations": {
            "is_family": True,
            "family": "postal_code",
            "rule_family": "country-specific pattern; read as text (keep leading zeros)",
            "role": "code",
            "head_tokens": ["code"],
            "entities": [],
        },
    },
    {
        "concept_name": "Year",
        "category": "Family",
        "description": "Generic calendar year value.",
        "aliases": ["year"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "year": True,
            "is_family": True,
            "family": "year",
            "rule_family": "domain range (e.g. 1900..current year+1); integer",
            "role": "time",
            "head_tokens": ["year"],
            "entities": [],
        },
    },
    {
        "concept_name": "Month",
        "category": "Family",
        "description": "Generic calendar month number from 1 to 12.",
        "aliases": ["month"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "integer_valued": True,
            "is_family": True,
            "family": "month",
            "rule_family": "domain range 1..12",
            "role": "time",
            "head_tokens": ["month"],
            "entities": [],
            "domain_range": [1, 12],
        },
    },
    {
        "concept_name": "Time Of Day",
        "category": "Family",
        "description": "Generic time of day such as 11:50:00, or an hour number from 0 to 23.",
        "aliases": ["time of day", "time", "hour"],
        "expected_data_types": ["string", "int", "datetime"],
        "profile_expectations": {
            "is_family": True,
            "family": "time_of_day",
            "rule_family": "time format HH:MM[:SS]; range 00:00-23:59",
            "role": "time",
            "head_tokens": ["time", "hour"],
            "entities": [],
        },
    },
    {
        "concept_name": "Region State",
        "category": "Family",
        "description": "Generic state, province or region name.",
        "aliases": ["state", "province"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "is_family": True,
            "family": "region_state",
            "rule_family": "allowed values from reference list; consistency with country",
            "role": "categorical",
            "head_tokens": [],
            "entities": [],
        },
    },
    {
        "concept_name": "Country",
        "category": "Family",
        "description": "Generic country name.",
        "aliases": ["country", "nation"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "is_family": True,
            "family": "country",
            "rule_family": "allowed values from reference list (ISO names)",
            "role": "categorical",
            "head_tokens": [],
            "entities": [],
        },
    },
    {
        "concept_name": "Country Code",
        "category": "Family",
        "description": "Generic country code such as ISO 3166 alpha-2 or alpha-3 codes.",
        "aliases": ["country code", "country iso code"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "is_family": True,
            "family": "country_code",
            "rule_family": "ISO 3166 alpha-2/alpha-3 allowed values",
            "role": "code",
            "head_tokens": ["code"],
            "entities": [],
            "value_regex": r"^[A-Za-z]{2,3}$",
        },
    },
    {
        "concept_name": "Currency Code",
        "category": "Family",
        "description": "Generic currency code such as the ISO 4217 codes USD, EUR or INR.",
        "aliases": ["currency code", "currency"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "is_family": True,
            "family": "currency_code",
            "rule_family": "ISO 4217 allowed values",
            "role": "code",
            "head_tokens": ["code", "currency"],
            "entities": [],
            "value_regex": r"^[A-Za-z]{3}$",
        },
    },
    {
        "concept_name": "Score Rating",
        "category": "Family",
        "description": "Generic bounded score or rating on a user-confirmed scale, such as 1 to 5 stars.",
        "aliases": ["score", "rating", "stars"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "is_family": True,
            "family": "score_rating",
            "rule_family": "bounded domain range from observed scale (user-confirmed, e.g. 1..5)",
            "role": "measure",
            "head_tokens": ["score", "rating"],
            "entities": [],
        },
    },
    {
        "concept_name": "Age",
        "category": "Family",
        "description": "Generic age of a person in years.",
        "aliases": ["age"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "bounded": True,
            "is_family": True,
            "family": "age",
            "rule_family": "domain range 0..120; integer",
            "role": "measure",
            "head_tokens": ["age"],
            "entities": [],
        },
    },
    {
        "concept_name": "Email",
        "category": "Family",
        "description": "Generic email address.",
        "aliases": ["email", "email address", "mail"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "email_pattern": True,
            "is_family": True,
            "family": "email",
            "rule_family": "email regex",
            "role": "text",
            "head_tokens": ["email"],
            "entities": [],
        },
    },
    {
        "concept_name": "Phone",
        "category": "Family",
        "description": "Generic telephone number, optionally stored as digits.",
        "aliases": ["phone", "phone number", "telephone", "mobile"],
        "expected_data_types": ["string", "int"],
        "profile_expectations": {
            "is_family": True,
            "family": "phone",
            "rule_family": "phone pattern (digits, separators, country code)",
            "role": "text",
            "head_tokens": ["phone", "telephone"],
            "entities": [],
        },
    },
    {
        "concept_name": "Address",
        "category": "Family",
        "description": "Generic street or mailing address.",
        "aliases": ["address", "street address", "mailing address"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "is_family": True,
            "family": "address",
            "rule_family": "completeness; consistency with city/postal",
            "role": "text",
            "head_tokens": ["address"],
            "entities": [],
        },
    },
    {
        "concept_name": "URL",
        "category": "Family",
        "description": "Generic web URL such as a product page or referer link.",
        "aliases": ["url", "link", "web address"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "url_pattern": True,
            "is_family": True,
            "family": "url",
            "rule_family": "URL pattern",
            "role": "text",
            "head_tokens": ["url", "link"],
            "entities": [],
        },
    },
    {
        "concept_name": "Percentage",
        "category": "Family",
        "description": "Generic percentage value on a 0 to 100 (or 0 to 1) range.",
        "aliases": ["percentage", "percent", "share"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "is_family": True,
            "family": "percentage",
            "rule_family": "domain range 0..100 (or 0..1 by observed unit)",
            "role": "measure",
            "head_tokens": ["percentage", "percent", "rate"],
            "entities": [],
        },
    },
    {
        "concept_name": "Ratio Metric",
        "category": "Family",
        "description": "Generic ratio or index metric on a bounded range confirmed by the user.",
        "aliases": ["ratio", "index", "ratio metric"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "is_family": True,
            "family": "ratio_metric",
            "rule_family": "numeric; bounded range (user-confirmed)",
            "role": "measure",
            "head_tokens": ["rate", "ratio", "index", "level"],
            "entities": [],
        },
    },
    {
        "concept_name": "Duration",
        "category": "Family",
        "description": "Generic duration or elapsed-time amount such as hours, days or years.",
        "aliases": ["duration", "elapsed time"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "is_family": True,
            "family": "duration",
            "rule_family": "numeric; non-negative",
            "role": "measure",
            "head_tokens": ["hours", "days", "years", "duration"],
            "entities": [],
        },
    },
    {
        "concept_name": "Latitude",
        "category": "Family",
        "description": "Generic geographic latitude in the range -90 to 90.",
        "aliases": ["latitude", "lat"],
        "expected_data_types": ["float", "int"],
        "profile_expectations": {
            "numeric": True,
            "is_family": True,
            "family": "latitude",
            "rule_family": "domain range -90..90",
            "role": "measure",
            "head_tokens": ["latitude", "lat"],
            "entities": [],
            "domain_range": [-90, 90],
        },
    },
    {
        "concept_name": "Longitude",
        "category": "Family",
        "description": "Generic geographic longitude in the range -180 to 180.",
        "aliases": ["longitude", "long", "lng"],
        "expected_data_types": ["float", "int"],
        "profile_expectations": {
            "numeric": True,
            "is_family": True,
            "family": "longitude",
            "rule_family": "domain range -180..180",
            "role": "measure",
            "head_tokens": ["longitude", "long", "lng"],
            "entities": [],
            "domain_range": [-180, 180],
        },
    },
    {
        "concept_name": "Geo Area",
        "category": "Family",
        "description": "Generic named geographic area such as a neighbourhood or sales region.",
        "aliases": ["geo area", "area name", "neighbourhood", "district"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "is_family": True,
            "family": "geo_area",
            "rule_family": "allowed values (observed, user-approved)",
            "role": "categorical",
            "head_tokens": [],
            "entities": [],
        },
    },
    {
        "concept_name": "Gender",
        "category": "Family",
        "description": "Generic gender or sex classification of a person.",
        "aliases": ["gender", "sex"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "is_family": True,
            "family": "gender",
            "rule_family": "allowed values (observed, user-approved)",
            "role": "categorical",
            "head_tokens": [],
            "entities": [],
            "value_vocabulary": ["male", "female", "m", "f", "other", "non-binary"],
        },
    },
    {
        "concept_name": "Other",
        "category": "Family",
        "description": "Generic fallback family for a column that carries data but matches no known family.",
        "aliases": ["other", "unknown"],
        "expected_data_types": ["string", "int", "float"],
        "profile_expectations": {
            "is_family": True,
            "family": "other",
            "rule_family": "ask user",
            "role": "other",
            "head_tokens": [],
            "entities": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Specific concepts added from the DEV split of the gold-label file only
# (Part G.3 rule: distinct reusable business meaning AND >= 2 datasets, or
# universally standard). Original 34 concept names are unchanged; their
# profile_expectations gain family metadata (Part G.2) and lose "unique".
# ---------------------------------------------------------------------------
ADDITIONAL_CONCEPTS = [
    {
        "concept_name": "Product Category",
        "category": "Classification",
        "description": "Category or grouping a product belongs to, such as electronics or groceries.",
        "aliases": ["product category", "product category name", "item category", "category name"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "categorical",
            "rule_family": "allowed values (observed, user-approved); case/whitespace variants",
            "role": "categorical",
            "head_tokens": [],
            "entities": ["category"],
        },
    },
    {
        "concept_name": "Brand Name",
        "category": "Product Attribute",
        "description": "Brand or manufacturer name of a product.",
        "aliases": ["brand name", "brand", "manufacturer"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "categorical",
            "rule_family": "allowed values (observed, user-approved); case/whitespace variants",
            "role": "categorical",
            "head_tokens": [],
            "entities": ["brand"],
        },
    },
    {
        "concept_name": "Payment Method",
        "category": "Business Attribute",
        "description": "Method used to pay for an order or transaction, such as credit card or voucher.",
        "aliases": ["payment method", "payment type", "payment"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "categorical",
            "rule_family": "allowed values (observed, user-approved); case/whitespace variants",
            "role": "categorical",
            "head_tokens": [],
            "entities": ["payment"],
        },
    },
    {
        "concept_name": "State Province",
        "category": "Geographic",
        "description": "State or province part of a geographic address.",
        "aliases": ["state", "province", "state province"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "region_state",
            "rule_family": "allowed values from reference list; consistency with country",
            "role": "categorical",
            "head_tokens": [],
            "entities": ["state", "province"],
        },
    },
    {
        "concept_name": "Latitude",
        "category": "Geographic",
        "description": "Geographic latitude of a location in decimal degrees (-90 to 90).",
        "aliases": ["latitude", "lat", "geo latitude"],
        "expected_data_types": ["float", "int"],
        "profile_expectations": {
            "numeric": True,
            "family": "latitude",
            "rule_family": "domain range -90..90",
            "role": "measure",
            "head_tokens": ["latitude", "lat"],
            "entities": [],
            "domain_range": [-90, 90],
        },
    },
    {
        "concept_name": "Longitude",
        "category": "Geographic",
        "description": "Geographic longitude of a location in decimal degrees (-180 to 180).",
        "aliases": ["longitude", "long", "lng", "geo longitude"],
        "expected_data_types": ["float", "int"],
        "profile_expectations": {
            "numeric": True,
            "family": "longitude",
            "rule_family": "domain range -180..180",
            "role": "measure",
            "head_tokens": ["longitude", "long", "lng"],
            "entities": [],
            "domain_range": [-180, 180],
        },
    },
    {
        "concept_name": "Employee Identifier",
        "category": "Identifier",
        "description": "Identifier assigned to an employee. Uniqueness is expected only for an approved primary key.",
        "aliases": ["employee id", "employee identifier", "staff id", "emp id", "employee number"],
        "expected_data_types": ["int", "string"],
        "profile_expectations": {
            "identifier_like": True,
            "non_null": True,
            "family": "identifier",
            "rule_family": "pattern (shape/regex); uniqueness only if approved PK; referential integrity if FK",
            "role": "identifier",
            "head_tokens": ["identifier", "code", "number", "id", "reference", "ref"],
            "entities": ["employee"],
        },
    },
    {
        "concept_name": "Store Identifier",
        "category": "Identifier",
        "description": "Identifier assigned to a store or shop location. Uniqueness is expected only for an approved primary key.",
        "aliases": ["store id", "store identifier", "store number", "shop id"],
        "expected_data_types": ["int", "string"],
        "profile_expectations": {
            "identifier_like": True,
            "non_null": True,
            "family": "identifier",
            "rule_family": "pattern (shape/regex); uniqueness only if approved PK; referential integrity if FK",
            "role": "identifier",
            "head_tokens": ["identifier", "code", "number", "id", "reference", "ref"],
            "entities": ["store"],
        },
    },
    {
        "concept_name": "Review Score",
        "category": "Measure",
        "description": "Numeric review or product rating on a bounded scale, typically 1 to 5.",
        "aliases": ["review score", "review rating", "product rating", "score"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "family": "score_rating",
            "rule_family": "bounded domain range from observed scale (user-confirmed, e.g. 1..5)",
            "role": "measure",
            "head_tokens": ["score", "rating"],
            "entities": ["review"],
        },
    },
    {
        "concept_name": "Discount Percentage",
        "category": "Financial Measure",
        "description": "Discount expressed as a percentage of the original price.",
        "aliases": ["discount percentage", "discount percent", "discount pct", "discount rate"],
        "expected_data_types": ["int", "float", "string"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "family": "percentage",
            "rule_family": "domain range 0..100 (or 0..1 by observed unit)",
            "role": "measure",
            "head_tokens": ["percentage", "percent"],
            "entities": ["discount"],
        },
    },
    {
        "concept_name": "Revenue",
        "category": "Financial Measure",
        "description": "Revenue generated by a business, account, customer, or transaction.",
        "aliases": ["revenue", "sales", "turnover", "sales revenue", "sales amount"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "continuous": True,
            "family": "money",
            "rule_family": "numeric; non-negative (user confirms: refunds); decimals <=2; outliers as anomaly",
            "role": "measure",
            "head_tokens": ["revenue", "sales", "amount"],
            "entities": [],
        },
    },
]

CONCEPTS = [
    {
        "concept_id": "C001",
        "concept_name": "Customer ID",
        "category": "Identifier",
        "description": "Identifier assigned to a customer. Uniqueness belongs to the approved primary key of a table, not to every customer reference (foreign keys legitimately repeat).",
        "aliases": ["customer_id", "customer number", "cust_id", "cust_no"],
        "expected_data_types": ["int", "string"],
        "profile_expectations": {
            "identifier_like": True,
            "non_null": True,
            "family": "identifier",
            "rule_family": "pattern (shape/regex); uniqueness only if approved PK; referential integrity if FK",
            "role": "identifier",
            "head_tokens": ["identifier", "code", "number", "id", "reference", "ref"],
            "entities": ["customer"],
        },
    },
    {
        "concept_id": "C002",
        "concept_name": "Account ID",
        "category": "Identifier",
        "description": "Identifier assigned to a business or customer account. Uniqueness belongs to the approved primary key of a table.",
        "aliases": ["account_id", "account number", "account code", "account_no"],
        "expected_data_types": ["int", "string"],
        "profile_expectations": {
            "identifier_like": True,
            "non_null": True,
            "family": "identifier",
            "rule_family": "pattern (shape/regex); uniqueness only if approved PK; referential integrity if FK",
            "role": "identifier",
            "head_tokens": ["identifier", "code", "number", "id", "reference", "ref"],
            "entities": ["account"],
        },
    },
    {
        "concept_id": "C003",
        "concept_name": "Product ID",
        "category": "Identifier",
        "description": "Identifier assigned to a product. Uniqueness belongs to the approved primary key of a table.",
        "aliases": ["product_id", "product code", "product number", "sku"],
        "expected_data_types": ["int", "string"],
        "profile_expectations": {
            "identifier_like": True,
            "non_null": True,
            "family": "identifier",
            "rule_family": "pattern (shape/regex); uniqueness only if approved PK; referential integrity if FK",
            "role": "identifier",
            "head_tokens": ["identifier", "code", "number", "id", "reference", "ref"],
            "entities": ["product"],
        },
    },
    {
        "concept_id": "C004",
        "concept_name": "Order ID",
        "category": "Identifier",
        "description": "Identifier assigned to a customer order or transaction. Uniqueness belongs to the approved primary key of a table.",
        "aliases": ["order_id", "order number", "order code", "order_no", "transaction id"],
        "expected_data_types": ["int", "string"],
        "profile_expectations": {
            "identifier_like": True,
            "non_null": True,
            "family": "identifier",
            "rule_family": "pattern (shape/regex); uniqueness only if approved PK; referential integrity if FK",
            "role": "identifier",
            "head_tokens": ["identifier", "code", "number", "id", "reference", "ref"],
            "entities": ["order", "transaction"],
        },
    },
    {
        "concept_id": "C005",
        "concept_name": "Revenue",
        "category": "Financial Measure",
        "description": "Revenue generated by a business, account, customer, or transaction.",
        "aliases": ["revenue", "sales", "turnover", "sales revenue", "sales amount"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "continuous": True,
            "family": "money",
            "rule_family": "numeric; non-negative (user confirms: refunds); decimals <=2; outliers as anomaly",
            "role": "measure",
            "head_tokens": ["revenue", "sales", "amount"],
            "entities": [],
        },
    },
    {
        "concept_id": "C006",
        "concept_name": "Price",
        "category": "Financial Measure",
        "description": "Monetary price associated with a product, item, or service.",
        "aliases": ["price", "unit price", "selling price", "item price", "fare"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "family": "money",
            "rule_family": "numeric; non-negative (user confirms: refunds); decimals <=2; outliers as anomaly",
            "role": "measure",
            "head_tokens": ["price", "cost", "fee", "fare", "amount"],
            "entities": [],
        },
    },
    {
        "concept_id": "C007",
        "concept_name": "Quantity",
        "category": "Measure",
        "description": "Number of units, items, or products associated with a transaction.",
        "aliases": ["quantity", "qty", "units", "item count"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "family": "count",
            "rule_family": "integer; non-negative; outliers as anomaly",
            "role": "count",
            "head_tokens": ["quantity", "count", "number"],
            "entities": [],
        },
    },
    {
        "concept_id": "C008",
        "concept_name": "Employee Count",
        "category": "Business Measure",
        "description": "Number of employees working for an organization or business.",
        "aliases": ["employees", "employee count", "number of employees", "staff count"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "family": "count",
            "rule_family": "integer; non-negative; outliers as anomaly",
            "role": "count",
            "head_tokens": ["count", "number"],
            "entities": ["employee"],
        },
    },
    {
        "concept_id": "C009",
        "concept_name": "Email",
        "category": "Contact Information",
        "description": "Electronic mail address used to contact a person or organization.",
        "aliases": ["email", "email address", "mail address"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "email_pattern": True,
            "family": "email",
            "rule_family": "email regex",
            "role": "text",
            "head_tokens": ["email"],
            "entities": [],
        },
    },
    {
        "concept_id": "C010",
        "concept_name": "Phone Number",
        "category": "Contact Information",
        "description": "Telephone number used to contact a person or organization.",
        "aliases": ["phone", "phone number", "mobile", "telephone"],
        "expected_data_types": ["string", "int"],
        "profile_expectations": {
            "phone_pattern": True,
            "family": "phone",
            "rule_family": "phone pattern (digits, separators, country code)",
            "role": "text",
            "head_tokens": ["phone", "telephone"],
            "entities": [],
        },
    },
    {
        "concept_id": "C011",
        "concept_name": "Date",
        "category": "Temporal",
        "description": "Calendar date representing when an event, transaction, or record occurred.",
        "aliases": ["date", "calendar date", "record date"],
        "expected_data_types": ["date", "datetime", "string"],
        "profile_expectations": {
            "temporal": True,
            "family": "date",
            "rule_family": "detected date format; parse rate; future/past plausibility; ordering vs other dates",
            "role": "date",
            "head_tokens": ["date", "day"],
            "entities": [],
        },
    },
    {
        "concept_id": "C012",
        "concept_name": "Age",
        "category": "Demographic",
        "description": "Age of a person, customer, passenger, or employee.",
        "aliases": ["age", "person age", "customer age"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "bounded": True,
            "family": "age",
            "rule_family": "domain range 0..120; integer",
            "role": "measure",
            "head_tokens": ["age"],
            "entities": [],
        },
    },
    {
        "concept_id": "C013",
        "concept_name": "Gender",
        "category": "Demographic",
        "description": "Gender or sex classification associated with a person.",
        "aliases": ["gender", "sex"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "gender",
            "rule_family": "allowed values (observed, user-approved)",
            "role": "categorical",
            "head_tokens": [],
            "entities": [],
            "value_vocabulary": ["male", "female", "m", "f", "other", "non-binary"],
        },
    },
    {
        "concept_id": "C014",
        "concept_name": "Country",
        "category": "Geographic",
        "description": "Country associated with a person, organization, address, or transaction.",
        "aliases": ["country", "country name", "nation"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "country",
            "rule_family": "allowed values from reference list (ISO names)",
            "role": "categorical",
            "head_tokens": [],
            "entities": [],
        },
    },
    {
        "concept_id": "C015",
        "concept_name": "City",
        "category": "Geographic",
        "description": "City or locality associated with a person, organization, or address.",
        "aliases": ["city", "city name", "town", "locality"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "city",
            "rule_family": "reference list + consistency city->state->country",
            "role": "categorical",
            "head_tokens": [],
            "entities": [],
        },
    },
    {
        "concept_id": "C016",
        "concept_name": "Postal Code",
        "category": "Geographic",
        "description": "Postal or ZIP code associated with a geographic address. Read as text to keep leading zeros.",
        "aliases": ["postal code", "zip code", "zipcode", "pin code", "zip"],
        "expected_data_types": ["string", "int"],
        "profile_expectations": {
            "postal_pattern": True,
            "family": "postal_code",
            "rule_family": "country-specific pattern; read as text (keep leading zeros)",
            "role": "code",
            "head_tokens": ["code"],
            "entities": [],
        },
    },
    {
        "concept_id": "C017",
        "concept_name": "Status",
        "category": "Business Attribute",
        "description": "Current state or status of a business entity, record, transaction, or process.",
        "aliases": ["status", "state", "current status", "record status", "stock state"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "categorical",
            "rule_family": "allowed values (observed, user-approved); case/whitespace variants",
            "role": "categorical",
            "head_tokens": ["status"],
            "entities": [],
        },
    },
    {
        "concept_id": "C018",
        "concept_name": "Address",
        "category": "Geographic",
        "description": "Physical or mailing address associated with a person, organization, or location.",
        "aliases": ["address", "street address", "mailing address", "office address"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "family": "address",
            "rule_family": "completeness; consistency with city/postal",
            "role": "text",
            "head_tokens": ["address"],
            "entities": [],
        },
    },
    {
        "concept_id": "C019",
        "concept_name": "Company Name",
        "category": "Business Attribute",
        "description": "Name of a company, organization, or business entity.",
        "aliases": ["company", "company name", "organization", "business name", "account name"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "family": "organization_name",
            "rule_family": "free text; completeness; duplicates (consistency)",
            "role": "text",
            "head_tokens": ["name"],
            "entities": ["company", "account"],
        },
    },
    {
        "concept_id": "C031",
        "concept_name": "Parent Company",
        "category": "Business Relationship",
        "description": "Name or identifier of the parent company or organization that owns or controls another company or subsidiary.",
        "aliases": [
            "parent company",
            "parent organization",
            "holding company",
            "subsidiary parent",
            "owned by",
            "parent",
        ],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "family": "organization_name",
            "rule_family": "free text; completeness; duplicates (consistency)",
            "role": "text",
            "head_tokens": ["name"],
            "entities": ["company", "parent"],
        },
    },
    {
        "concept_id": "C020",
        "concept_name": "Industry Sector",
        "category": "Business Attribute",
        "description": "Industry or business sector in which an organization operates.",
        "aliases": ["sector", "industry", "industry sector", "business sector"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "categorical",
            "rule_family": "allowed values (observed, user-approved); case/whitespace variants",
            "role": "categorical",
            "head_tokens": [],
            "entities": ["industry", "sector"],
        },
    },
    {
        "concept_id": "C021",
        "concept_name": "Year Established",
        "category": "Business Attribute",
        "description": "Year in which a company or organization was established.",
        "aliases": ["year established", "founded year", "established year", "foundation year", "year built"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "year": True,
            "family": "year",
            "rule_family": "domain range (e.g. 1900..current year+1); integer",
            "role": "time",
            "head_tokens": ["year"],
            "entities": ["established", "founded", "built", "construction"],
        },
    },
    {
        "concept_id": "C022",
        "concept_name": "Transaction Date",
        "category": "Temporal",
        "description": "Date or timestamp when a financial or business transaction occurred.",
        "aliases": ["transaction date", "txn date", "purchase date", "order date"],
        "expected_data_types": ["date", "datetime", "string"],
        "profile_expectations": {
            "temporal": True,
            "family": "date",
            "rule_family": "detected date format; parse rate; future/past plausibility; ordering vs other dates",
            "role": "date",
            "head_tokens": ["date"],
            "entities": ["transaction", "purchase", "order"],
        },
    },
    {
        "concept_id": "C023",
        "concept_name": "Customer Name",
        "category": "Person / Customer",
        "description": "Name of a customer or individual associated with a customer record.",
        "aliases": ["customer name", "customer", "client name", "client"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "family": "person_name",
            "rule_family": "pattern: letters/spaces/punctuation; no digits; completeness",
            "role": "text",
            "head_tokens": ["name"],
            "entities": ["customer", "client"],
        },
    },
    {
        "concept_id": "C024",
        "concept_name": "Product Name",
        "category": "Product Attribute",
        "description": "Name or descriptive label assigned to a product.",
        "aliases": ["product name", "product", "item name", "item"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "family": "product_name",
            "rule_family": "free text; completeness",
            "role": "text",
            "head_tokens": ["name"],
            "entities": ["product", "item"],
        },
    },
    {
        "concept_id": "C025",
        "concept_name": "Description",
        "category": "Text Attribute",
        "description": "Free-text description providing additional information about an entity or record.",
        "aliases": ["description", "details", "notes", "comments"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "family": "free_text",
            "rule_family": "completeness; max length; not validated by pattern",
            "role": "text",
            "head_tokens": ["description", "text", "details"],
            "entities": [],
        },
    },
    {
        "concept_id": "C026",
        "concept_name": "Category",
        "category": "Classification",
        "description": "Classification or grouping assigned to an entity, product, or record.",
        "aliases": ["category", "class", "classification", "group"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "categorical",
            "rule_family": "allowed values (observed, user-approved); case/whitespace variants",
            "role": "categorical",
            "head_tokens": [],
            "entities": ["category"],
        },
    },
    {
        "concept_id": "C027",
        "concept_name": "Discount",
        "category": "Financial Measure",
        "description": "Reduction applied to the original price or transaction amount.",
        "aliases": ["discount", "discount amount", "discount rate", "markdown"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "family": "money",
            "rule_family": "numeric; non-negative (user confirms: refunds); decimals <=2; outliers as anomaly",
            "role": "measure",
            "head_tokens": ["amount", "rate", "discount"],
            "entities": ["discount"],
        },
    },
    {
        "concept_id": "C028",
        "concept_name": "Tax",
        "category": "Financial Measure",
        "description": "Tax amount or tax rate associated with a transaction or financial record.",
        "aliases": ["tax", "tax amount", "tax rate", "sales tax"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "family": "money",
            "rule_family": "numeric; non-negative (user confirms: refunds); decimals <=2; outliers as anomaly",
            "role": "measure",
            "head_tokens": ["amount", "rate"],
            "entities": ["tax"],
        },
    },
    {
        "concept_id": "C029",
        "concept_name": "Currency",
        "category": "Financial Attribute",
        "description": "Currency in which a monetary value or transaction is expressed.",
        "aliases": ["currency", "currency code", "money type"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "currency_code": True,
            "family": "currency_code",
            "rule_family": "ISO 4217 allowed values",
            "role": "code",
            "head_tokens": ["currency", "code"],
            "entities": [],
            "value_regex": r"^[A-Za-z]{3}$",
        },
    },
    {
        "concept_id": "C030",
        "concept_name": "Review Text",
        "category": "Text Attribute",
        "description": "Free-text review or feedback provided by a customer about a product or service.",
        "aliases": ["review", "review text", "feedback", "customer review", "comment", "content"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "long_text": True,
            "family": "free_text",
            "rule_family": "completeness; max length; not validated by pattern",
            "role": "text",
            "head_tokens": ["text", "review", "content", "comment"],
            "entities": ["review"],
        },
    },
    {
        "concept_id": "C033",
        "concept_name": "Customer Type",
        "category": "Classification",
        "description": "Classification of a customer into a business category or segment.",
        "aliases": ["customer type", "type", "client type", "customer segment", "segment"],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "categorical": True,
            "family": "categorical",
            "rule_family": "allowed values (observed, user-approved); case/whitespace variants",
            "role": "categorical",
            "head_tokens": [],
            "entities": ["customer", "type", "segment"],
        },
    },
    {
        "concept_id": "C034",
        "concept_name": "Amount",
        "category": "Financial Measure",
        "description": "Monetary amount associated with a transaction, invoice, or record.",
        "aliases": ["amount", "total amount", "transaction amount", "amount value"],
        "expected_data_types": ["int", "float"],
        "profile_expectations": {
            "numeric": True,
            "non_negative": True,
            "family": "money",
            "rule_family": "numeric; non-negative (user confirms: refunds); decimals <=2; outliers as anomaly",
            "role": "measure",
            "head_tokens": ["amount", "total", "value"],
            "entities": [],
        },
    },
    {
        "concept_id": "C032",
        "concept_name": "Office Location",
        "category": "Geographic Information",
        "description": "Geographic location of the office, headquarters, or primary business location of an organization.",
        "aliases": [
            "office location",
            "office headquarters",
            "headquarters location",
            "business location",
            "company location",
            "hq location",
        ],
        "expected_data_types": ["string"],
        "profile_expectations": {
            "text": True,
            "categorical": True,
            "family": "country",
            "rule_family": "allowed values from reference list (ISO names)",
            "role": "categorical",
            "head_tokens": ["location"],
            "entities": ["office"],
        },
    },
]

# Families that duplicate a specific concept keep ONE KB row; the specific
# concept doubles as the family when its profile_expectations carry is_family.
# Families whose concept_name collides with a specific concept do NOT get
# their own KB row (concept_name is unique in the database); the specific
# concept doubles as the family (dual role) when expand_definitions merges
# the family flags into it.
def _specific_names() -> set[str]:
    return {
        concept["concept_name"] for concept in CONCEPTS
    } | {
        concept["concept_name"] for concept in ADDITIONAL_CONCEPTS
    }


def family_concepts() -> list[dict]:
    """Family definitions seeded as their OWN concept rows.

    Families whose name collides with a specific concept (City, Date, Age,
    Email, ...) are excluded here: that specific concept carries the family
    metadata instead (dual role).
    """
    specific = _specific_names()
    return [
        family
        for family in FAMILIES
        if family["concept_name"] not in specific
    ]


def _definition(concept_name: str) -> dict | None:
    for concept in CONCEPTS:
        if concept["concept_name"] == concept_name:
            return concept
    return None


def expand_definitions(baseline_only: bool = False) -> list[dict]:
    """Shipped KB definitions.

    ``baseline_only`` reproduces the pre-upgrade KB: the original 34 specific
    concepts WITHOUT family flags or additions (used by the evaluation
    harness to measure what the KB expansion buys). The full KB marks the
    dual-role concepts (families sharing a name with a specific concept) with
    ``is_family`` and appends the remaining families plus the dev-split
    specific additions.
    """
    definitions = [dict(concept) for concept in CONCEPTS]

    if baseline_only:
        return definitions

    family_by_name = {
        family["concept_name"]: family for family in FAMILIES
    }

    merged: list[dict] = []
    for concept in definitions:
        merged_concept = dict(concept)
        family = family_by_name.get(merged_concept["concept_name"])
        if family is not None:
            expectations = dict(merged_concept["profile_expectations"])
            expectations["is_family"] = True
            merged_concept["profile_expectations"] = expectations
        merged.append(merged_concept)

    existing_names = {concept["concept_name"] for concept in merged}

    for concept in ADDITIONAL_CONCEPTS:
        if concept["concept_name"] in existing_names:
            # Revenue: ADDITIONAL duplicates a baseline concept name; the
            # baseline definition (with concept_id) wins.
            continue
        merged_concept = dict(concept)
        family = family_by_name.get(merged_concept["concept_name"])
        if family is not None:
            expectations = dict(merged_concept["profile_expectations"])
            expectations["is_family"] = True
            merged_concept["profile_expectations"] = expectations
        merged.append(merged_concept)
        existing_names.add(merged_concept["concept_name"])

    for family in FAMILIES:
        if family["concept_name"] in existing_names:
            continue  # dual-role row already present
        standalone = dict(family)
        standalone["family_only"] = True
        merged.append(standalone)

    return merged


def ensure_semantic_kb(db: Session) -> None:
    """Idempotent one-call KB bootstrap used by Stage 04 runs.

    Seeds the shipped concepts (families + specific, refreshed when the
    code-defined baseline differs) and backfills any missing current-
    version embeddings. Cheap no-ops when the KB is already current.
    """
    seed_semantic_concepts(db)

    from app.services.semantic_retrieval import ensure_kb_embeddings

    ensure_kb_embeddings(db)


def seed_semantic_concepts(db: Session, baseline_only: bool = False) -> int:
    """Insert the shipped Semantic Knowledge Base and keep it in sync.

    Idempotent and safe to call on every analysis run (and on databases that
    already contain the original 34 concepts): existing concepts are
    refreshed when the code-defined baseline differs, new concepts are
    inserted, human alias admissions in separate KB-version tables are never
    touched. ``baseline_only`` seeds the original 34 concepts without the
    family layer (evaluation-harness baseline mode).
    """
    inserted = 0

    for item in expand_definitions(baseline_only=baseline_only):
        existing = (
            db.query(SemanticConcept)
            .filter(
                SemanticConcept.concept_name
                == item["concept_name"]
            )
            .first()
        )

        if existing:
            updated = False

            if existing.description != item["description"]:
                existing.description = item["description"]
                updated = True

            if existing.aliases != json.dumps(item["aliases"]):
                existing.aliases = json.dumps(item["aliases"])
                updated = True

            if existing.expected_data_types != json.dumps(
                item["expected_data_types"]
            ):
                existing.expected_data_types = json.dumps(
                    item["expected_data_types"]
                )
                updated = True

            if existing.profile_expectations != json.dumps(
                item["profile_expectations"]
            ):
                existing.profile_expectations = json.dumps(
                    item["profile_expectations"]
                )
                updated = True

            if updated:
                # The embedding text depends on the natural-language fields.
                from app.services.semantic_embedding import SemanticEmbeddingService

                SemanticEmbeddingService().create_or_update_embedding(
                    db, existing
                )

            continue

        concept = SemanticConcept(
            concept_name=item["concept_name"],
            category=item["category"],
            description=item["description"],
            aliases=json.dumps(item["aliases"]),
            expected_data_types=json.dumps(
                item["expected_data_types"]
            ),
            profile_expectations=json.dumps(
                item["profile_expectations"]
            ),
        )

        db.add(concept)
        inserted += 1

    db.commit()

    return inserted


# ---------------------------------------------------------------------------
# Glue-split vocabulary (Part A.4): the conservative glued-token splitter only
# accepts parts that are KNOWN words. Register the KB's generic vocabulary
# (alias tokens, family head tokens) so names like "warehouseid" or "orderdt"
# split into known words without any dataset-specific configuration.
# ---------------------------------------------------------------------------
def register_kb_vocabulary() -> None:
    """Expose the KB's generic vocabulary to the name normalizer."""
    from app.services.semantic_name import register_known_words

    words: set[str] = set()
    for concept in expand_definitions():
        words.add(concept["concept_name"].lower())
        for alias in concept["aliases"]:
            words.update(str(alias).lower().split())
        expectations = concept["profile_expectations"]
        words.update(str(token) for token in expectations.get("head_tokens", []))
    register_known_words(words)


register_kb_vocabulary()
