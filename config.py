"""
config.py
---------
Universe definition for the Trends-Alpha project.

Each ticker is mapped to a consumer-facing search keyword that plausibly
tracks demand/attention for that company. This is the single most important
research choice in the project -- a bad keyword (too generic, too niche,
contaminated by unrelated news) will kill the signal before any modeling
even starts.
"""

UNIVERSE = {
    "AAPL": "iPhone",
    "TSLA": "Tesla",
    "NFLX": "Netflix",
    "NKE":  "Nike Shoes",
    "SBUX": "Starbucks",
    "DIS":  "Disney Plus",
    "PTON": "Peloton",
    "ROKU": "Roku",
    "ABNB": "Airbnb",
    "CHWY": "Chewy",
    "ETSY": "Etsy",
    "W":    "Wayfair",
    "CROX": "Crocs",
    "DKNG": "DraftKings",
    "RH":   "RH Furniture",
    "LULU": "Lululemon",
}

# How far back to pull (Google Trends supports up to 5 years of weekly data
# in a single pull; going further requires stitching multiple overlapping
# pulls together and rescaling, which pytrends does not do automatically).
TIMEFRAME = "today 5-y"
GEO = "US"  # restrict to US search interest; leave "" for worldwide

# Rolling window (in weeks) used to z-score each keyword's own trend history
Z_WINDOW = 26

# Number of quantile buckets for the long/short portfolio
N_QUANTILES = 5

# Trading periods per year, for annualizing weekly return statistics
PERIODS_PER_YEAR = 52
