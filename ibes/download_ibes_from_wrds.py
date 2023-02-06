# %%
# Packages
import wrds
import pandas as pd



# %%
# Read iclink.csv
iclink = pd.read_csv("../03_data/iclink.csv")
iclink = iclink.rename(columns={"TICKER": "ticker", "PERMNO": "permno", "SCORE": "score"})
iclink = iclink[["ticker", "permno", "score"]]
iclink = iclink.sort_values("score")
iclink = iclink.drop_duplicates(subset=["ticker"], keep="first")


# %%
# Open database connection
db = wrds.Connection(wrds_username="twied03")


# Download announcement dates
query = """
        SELECT DISTINCT     ticker,
                            cusip,
                            anndats_act
        FROM 
            tr_ibes.statsum_epsus
    """
an_dates = db.raw_sql(query)
an_dates["anndats_act"] = pd.to_datetime(
    an_dates["anndats_act"], format="%Y-%m-%d")

# Merge with iclink
an_dates = an_dates.merge(iclink, on="ticker")

# Save
an_dates = an_dates.dropna(subset=["permno", "anndats_act"])
an_dates.to_parquet("../03_data/ibes_announcement_dates.pq")


# Close database connection
db.close()



