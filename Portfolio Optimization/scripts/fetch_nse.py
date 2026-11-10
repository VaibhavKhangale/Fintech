"""Archive NSE public data locally. Safe to rerun: finished items are skipped."""
import argparse
from datetime import date

from pmo import nse

START = date(2010, 6, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["bhavcopy", "api"])
    args = parser.parse_args()
    today = date.today()
    if args.what == "bhavcopy":
        print("stored", nse.download_bhavcopies(START, today), flush=True)
        return
    stored = nse.download_monthly("corporates-corporateActions?index=equities", nse.ACTIONS_DIR,
                                  START, today)
    print("action months", stored, flush=True)
    stored = nse.download_monthly(
        "corporates-financial-results?index=equities&period=Quarterly", nse.RESULTS_DIR,
        date(2011, 1, 1), today)
    print("result months", stored, flush=True)
    nse.fetch_symbol_changes().to_csv(nse.BHAV_DIR.parent / "symbol_changes.csv", index=False)
    print("symbol changes saved", flush=True)


if __name__ == "__main__":
    main()
