"""
Data Wrangling Using Book Dataset — Cleaning, Transformation & Visualization
Input : my_openlibrary_books_data.csv  (collected with api_call.py + data_collection.py)
Output: books_cleaned.csv, wrangling_log.txt, charts/*.png
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dateutil import parser as dparser

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "my_openlibrary_books_data.csv")
OUT_CSV = os.path.join(BASE, "books_cleaned.csv")
LOG = os.path.join(BASE, "wrangling_log.txt")
CHART_DIR = os.path.join(BASE, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

log_lines = []
fixes = {}
def log(msg=""):
    print(msg)
    log_lines.append(msg)

# =====================================================================
# 1. LOAD RAW DATA
# =====================================================================
raw = pd.read_csv(RAW)
df = raw.copy()
log("=" * 64)
log("STEP 1 — RAW DATA")
log("=" * 64)
log(f"Rows: {len(df)}   Columns: {df.shape[1]}")
log("Missing values per column (raw):")
for c, n in df.isna().sum().items():
    if n:
        log(f"   {c:<26}{n:>5}  ({n/len(df):.0%})")
missing_before = df.isna().mean() * 100

# =====================================================================
# 2. FIX FORMATTING — whitespace, author names
# =====================================================================
log("\nSTEP 2 — FIX FORMATTING")
for col in ["Title", "Author", "Genre_Detail"]:
    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

def clean_authors(s):
    """Remove 'Adapted by …' notes and repeated names: 'Richelle Mead, Richelle Mead' -> 'Richelle Mead'."""
    names, seen = [], set()
    for n in s.split(","):
        n = re.sub(r"^(adapted|edited|translated|illustrated)\s+by\s+", "", n.strip(), flags=re.I)
        if n and n.lower() not in seen:
            seen.add(n.lower())
            names.append(n)
    return ", ".join(names) if names else "Unknown"

before = df["Author"].copy()
df["Author"] = df["Author"].apply(clean_authors)
fixes["Author names cleaned"] = int((before != df["Author"]).sum())
log(f"   Author strings cleaned (duplicates/'Adapted by' removed): {(before != df['Author']).sum()}")
df["Primary_Author"] = df["Author"].str.split(",").str[0].str.strip()
df["Author_Count"] = df["Author"].str.count(",") + 1
log(f"   New columns: Primary_Author, Author_Count (multi-author books: {(df['Author_Count']>1).sum()})")

# =====================================================================
# 3. STANDARDIZE GENRES  (snake_case keys -> readable labels)
# =====================================================================
log("\nSTEP 3 — STANDARDIZE GENRES")
genre_map = {
    "science_fiction": "Sci-Fi", "young_adult": "Young Adult",
    "historical_fiction": "Historical Fiction", "romance": "Romance",
    "mystery": "Mystery", "horror": "Horror", "thriller": "Thriller",
    "fantasy": "Fantasy",
}
df["Genre"] = df["Genre_Category"].map(genre_map).fillna(df["Genre_Category"].str.title())
log("   e.g. 'science_fiction' -> 'Sci-Fi', 'young_adult' -> 'Young Adult'")

# Genre_Detail subjects: title-case, drop duplicates inside a row, drop Open Library noise tags
noise = {"open library staff picks", "accessible book", "protected daisy", "in library", "large type books"}
def clean_subjects(s):
    out, seen = [], set()
    for t in s.split(","):
        t = t.strip()
        k = t.lower()
        if t and k not in seen and k not in noise:
            seen.add(k)
            out.append(t[:1].upper() + t[1:])
    return ", ".join(out)
df["Genre_Detail"] = df["Genre_Detail"].apply(clean_subjects)

# =====================================================================
# 4. DATES — mixed formats -> ISO (YYYY-MM-DD) + year column
#    raw had '13 November 1850', '1856-06-22', 'August 14, 1947', '1950', '11 Mar 1969'
# =====================================================================
log("\nSTEP 4 — STANDARDIZE DATES")
def to_iso(v):
    if not isinstance(v, str) or not v.strip():
        return np.nan, np.nan
    v = v.strip()
    if re.fullmatch(r"\d{4}", v):              # year only -> keep year, no fake day
        return v, int(v)
    try:
        d = dparser.parse(v, default=pd.Timestamp("1900-01-01"))
        return d.strftime("%Y-%m-%d"), d.year
    except (ValueError, OverflowError):
        m = re.search(r"(\d{4})", v)
        return (m.group(1), int(m.group(1))) if m else (np.nan, np.nan)

for col in ["Author_Birth_Date", "Author_Death_Date"]:
    parsed = df[col].apply(to_iso)
    n_formats = df[col].dropna().str.replace(r"\d", "9", regex=True).nunique()
    df[col] = parsed.str[0]
    df[col.replace("_Date", "_Year")] = pd.to_numeric(parsed.str[1], errors="coerce").astype("Int64")
    fixes[f"{col.split('_')[1]} dates re-formatted"] = int(df[col].notna().sum())
    log(f"   {col}: {n_formats} different raw patterns -> one ISO format")

# =====================================================================
# 5. REMOVE INCORRECT VALUES
# =====================================================================
log("\nSTEP 5 — REMOVE INCORRECT / IMPOSSIBLE VALUES")
df["Publish_Year"] = df["Publish_Year"].astype("Int64")

bad_year = df["Publish_Year"].lt(1450) | df["Publish_Year"].gt(2026)
log(f"   Publish_Year outside 1450–2026 (e.g. 0) -> set missing: {bad_year.sum()}")
df.loc[bad_year, "Publish_Year"] = pd.NA

# Open Library stores 1800 / 1900 as a "some time in that century" placeholder:
# 45 books in 1800 but 0 in 1799 and 1801; 'The Hound of the Baskervilles' (1902) and
# 'An American Tragedy' (1925) are listed as 1900.
placeholder = df["Publish_Year"].isin([1800, 1900])
log(f"   Placeholder years 1800/1900 (century guesses, not real dates) -> set missing: {placeholder.sum()}")
df.loc[placeholder, "Publish_Year"] = pd.NA
fixes["Invalid / placeholder years"] = int(bad_year.sum() + placeholder.sum())

# a book can't be published before its author was ~10 years old
impossible = df["Publish_Year"].notna() & df["Author_Birth_Year"].notna() & \
             (df["Publish_Year"] < df["Author_Birth_Year"] + 10)
log(f"   Publish_Year earlier than author's birth+10 -> set missing: {impossible.sum()}")
for _, r in df[impossible].iterrows():
    log(f"      · {r['Title']} ({r['Primary_Author']}): published {r['Publish_Year']}, author born {r['Author_Birth_Year']}")
df.loc[impossible, "Publish_Year"] = pd.NA
fixes["Year before author's birth"] = int(impossible.sum())

# death before birth
bad_life = df["Author_Death_Year"].notna() & df["Author_Birth_Year"].notna() & \
           (df["Author_Death_Year"] < df["Author_Birth_Year"])
df.loc[bad_life, ["Author_Death_Date", "Author_Death_Year"]] = pd.NA
log(f"   Death year before birth year -> removed: {bad_life.sum()}")

# ratings: Open Library returns average 0.0 when there are 0 ratings -> that is "no rating", not a score of 0
zero_r = df["Rating_Count"].eq(0)
df.loc[zero_r, ["Average_Rating", "Rating_Count"]] = np.nan
fixes["Fake 0.0 ratings"] = int(zero_r.sum())
log(f"   Average_Rating = 0 with 0 ratings -> set missing: {zero_r.sum()}")
df["Average_Rating"] = df["Average_Rating"].round(2)
out_range = ~df["Average_Rating"].between(1, 5) & df["Average_Rating"].notna()
df.loc[out_range, "Average_Rating"] = np.nan
log(f"   Average_Rating outside 1–5 -> set missing: {out_range.sum()}")

# page counts under 30 for novels are data-entry errors (e.g. 'Dumb Witness' = 1 page)
bad_pages = df["Page_Count"].lt(30)
log(f"   Page_Count < 30 -> set missing: {bad_pages.sum()}")
df.loc[bad_pages, "Page_Count"] = np.nan
fixes["Impossible page counts"] = int(bad_pages.sum())

# =====================================================================
# 6. REMOVE DUPLICATES — same title + same primary author = same book
#    (Open_Library_Key was unique, but the same novel appeared under two genres)
# =====================================================================
log("\nSTEP 6 — REMOVE DUPLICATES")
df["_t"] = df["Title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
df["_a"] = df["Primary_Author"].str.lower()
df["_filled"] = df.notna().sum(axis=1)
all_genres = df.groupby(["_t", "_a"])["Genre"].agg(lambda g: ", ".join(sorted(set(g))))
dups = df[df.duplicated(["_t", "_a"], keep=False)]
for (t, a), g in dups.groupby(["_t", "_a"]):
    log(f"   · '{g['Title'].iloc[0]}' by {g['Primary_Author'].iloc[0]} appeared {len(g)}x "
        f"({', '.join(g['Genre'])}) -> merged")
# keep the most complete record, remember every genre it appeared under
df = df.sort_values("_filled", ascending=False).drop_duplicates(["_t", "_a"], keep="first")
df["All_Genres"] = df.set_index(["_t", "_a"]).index.map(all_genres)
fixes["Duplicate books merged"] = int(len(dups) - dups.groupby(['_t','_a']).ngroups)
log(f"   Rows removed: {len(dups) - dups.groupby(['_t','_a']).ngroups}")
log("   Same title / different author (e.g. 'The Hollow' Christie vs Roberts) kept — different books.")
df = df.drop(columns=["_t", "_a", "_filled"])

# =====================================================================
# 7. HANDLE MISSING VALUES
# =====================================================================
log("\nSTEP 7 — HANDLE MISSING VALUES")
enriched = df["Want_To_Read_Count"].notna()
df["Is_Enriched"] = enriched
log(f"   Enrichment (ratings / shelves / author info) exists for {enriched.sum()} of {len(df)} books "
    f"({enriched.mean():.0%}). The enrichment script had not finished for the rest.")
log("   -> These are NOT filled with averages (that would invent data). Analyses on ratings/")
log("      engagement use only the enriched subset; flag column Is_Enriched added.")

# shelf counts: for enriched rows a missing count means 0 people shelved it
for c in ["Want_To_Read_Count", "Currently_Reading_Count", "Already_Read_Count"]:
    df.loc[enriched, c] = df.loc[enriched, c].fillna(0)
df["Author_Bio"] = df["Author_Bio"].fillna("Not available")
# author alive?  known birth + no death + born after 1900 -> living
df["Author_Status"] = np.select(
    [df["Author_Death_Year"].notna(),
     df["Author_Birth_Year"].notna() & (df["Author_Birth_Year"] > 1900)],
    ["Deceased", "Living"], default="Unknown")

# page count: fill with genre median ONLY where we have enough real values, and flag it
df["Page_Count_Imputed"] = False
med = df.groupby("Genre")["Page_Count"].median()
cnt = df.groupby("Genre")["Page_Count"].count()
fill_mask = df["Page_Count"].isna() & df["Is_Enriched"] & df["Genre"].map(cnt).ge(5)
df.loc[fill_mask, "Page_Count"] = df.loc[fill_mask, "Genre"].map(med)
df.loc[fill_mask, "Page_Count_Imputed"] = True
fixes["Page counts imputed (flagged)"] = int(fill_mask.sum())
fixes["Genres standardized"] = int(len(df))
log(f"   Page_Count: {fill_mask.sum()} enriched books filled with their genre median (flagged in Page_Count_Imputed)")

# =====================================================================
# 8. TRANSFORMATION — new analytical columns
# =====================================================================
log("\nSTEP 8 — TRANSFORMATION (new features)")
df["Decade"] = (df["Publish_Year"] // 10 * 10).astype("Int64")
df["Era"] = pd.cut(df["Publish_Year"].astype("float"),
                   bins=[1449, 1899, 1949, 1999, 2026],
                   labels=["Pre-1900", "1900–1949", "1950–1999", "2000+"])
df["Book_Age"] = (2026 - df["Publish_Year"]).astype("Int64")
df["Total_Readers"] = df[["Want_To_Read_Count", "Currently_Reading_Count", "Already_Read_Count"]].sum(axis=1, min_count=1)
df["Completion_Rate"] = (df["Already_Read_Count"] /
                         (df["Already_Read_Count"] + df["Currently_Reading_Count"]).replace(0, np.nan)).round(3)
df["Length_Category"] = pd.cut(df["Page_Count"], [0, 200, 350, 500, np.inf],
                               labels=["Short (<200)", "Medium (200–350)", "Long (350–500)", "Epic (500+)"])
df["Popularity_Tier"] = pd.cut(df["Edition_Count"], [0, 5, 25, 100, np.inf],
                               labels=["Niche", "Moderate", "Popular", "Classic/Bestseller"], include_lowest=True)
df["Author_Age_At_Publish"] = (df["Publish_Year"] - df["Author_Birth_Year"]).astype("Int64")
df.loc[~df["Author_Age_At_Publish"].between(10, 100), "Author_Age_At_Publish"] = pd.NA
log("   Added: Decade, Era, Book_Age, Total_Readers, Completion_Rate, Length_Category,")
log("          Popularity_Tier, Author_Age_At_Publish, Author_Status, Primary_Author, Author_Count")

# =====================================================================
# 9. SAVE READY DATASET
# =====================================================================
cols = ["Title", "Author", "Primary_Author", "Author_Count", "Genre", "All_Genres", "Genre_Detail",
        "Publish_Year", "Decade", "Era", "Book_Age", "Edition_Count", "Popularity_Tier", "Ebook_Available",
        "Average_Rating", "Rating_Count", "Page_Count", "Page_Count_Imputed", "Length_Category",
        "Want_To_Read_Count", "Currently_Reading_Count", "Already_Read_Count", "Total_Readers",
        "Completion_Rate", "Author_Birth_Date", "Author_Birth_Year", "Author_Death_Date", "Author_Death_Year",
        "Author_Status", "Author_Age_At_Publish", "Author_Bio", "Is_Enriched", "Open_Library_Key"]
df = df[cols].sort_values(["Genre", "Title"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

log("\n" + "=" * 64)
log("READY DATASET")
log("=" * 64)
log(f"Rows: {len(raw)} -> {len(df)}   Columns: {raw.shape[1]} -> {df.shape[1]}")

# =====================================================================
# 10. VISUALIZATION  (colours match the presentation theme)
# =====================================================================
BG, PANEL, GRID = "#0a1230", "#121d45", "#26325e"
CYAN, BLUE, INDIGO, SKY, GREEN, PINK = "#00d4ff", "#2f6df6", "#6366f1", "#7fe3ff", "#34e3a8", "#ff5c8a"
PALETTE = [CYAN, BLUE, INDIGO, SKY, GREEN, "#a78bfa", "#38bdf8", "#f59e0b"]
TXT, SUB = "#ffffff", "#9fb3d9"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB,
    "text.color": TXT, "axes.titlecolor": CYAN, "axes.titleweight": "bold", "axes.titlesize": 15,
    "axes.titlepad": 14, "font.size": 11, "grid.color": GRID, "axes.grid": True, "grid.alpha": .6,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})
def save(fig, name, note=None):
    if note:
        fig.text(0.01, 0.01, note, fontsize=8.5, color=SUB, style="italic")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(os.path.join(CHART_DIR, name), dpi=180)
    plt.close(fig)

E = df[df["Is_Enriched"]]
genre_order = df["Genre"].value_counts().index

# 1 — Books by genre
fig, ax = plt.subplots(figsize=(10, 5.6))
vc = df["Genre"].value_counts()
bars = ax.bar(vc.index, vc.values, color=CYAN, width=.65)
ax.bar_label(bars, padding=3, color=TXT, fontsize=10)
ax.set_title("Books by Genre"); ax.set_ylabel("Number of books"); ax.grid(axis="x", visible=False)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
save(fig, "01_books_by_genre.png", f"n = {len(df)} books after cleaning")

# 2 — Genre distribution (donut)
fig, ax = plt.subplots(figsize=(8, 6.2))
w, t, a = ax.pie(vc.values, labels=vc.index, colors=PALETTE, autopct="%1.1f%%", startangle=90,
                 pctdistance=.8, wedgeprops=dict(width=.38, edgecolor=BG, linewidth=2),
                 textprops=dict(color=TXT, fontsize=10))
for x in a: x.set_color(BG); x.set_fontweight("bold"); x.set_fontsize(9)
ax.text(0, 0, f"{len(df)}\nbooks", ha="center", va="center", fontsize=16, fontweight="bold", color=TXT)
ax.set_title("Genre Distribution")
save(fig, "02_genre_distribution.png")

# 3 — Books published per decade (replaces 'sales by year' — no sales data in Open Library)
fig, ax = plt.subplots(figsize=(11, 5.6))
dec = df[df["Decade"] >= 1800]["Decade"].value_counts().sort_index()
ax.plot(dec.index.astype(int), dec.values, color=CYAN, lw=2.6, marker="o", ms=5)
ax.fill_between(dec.index.astype(int), dec.values, color=CYAN, alpha=.12)
ax.set_title("Books Published per Decade (1800 onwards)"); ax.set_xlabel("Decade"); ax.set_ylabel("Books")
save(fig, "03_books_per_decade.png", f"{int(df['Decade'].lt(1800).sum())} pre-1800 books and "
     f"{int(df['Decade'].isna().sum())} with invalid/missing year not shown.  2020s is a partial decade.")

# 4 — Genre mix by era (stacked %)
fig, ax = plt.subplots(figsize=(11, 5.6))
ct = pd.crosstab(df["Era"], df["Genre"], normalize="index")[genre_order] * 100
left = np.zeros(len(ct))
for i, g in enumerate(ct.columns):
    ax.barh(ct.index.astype(str), ct[g], left=left, color=PALETTE[i], label=g, edgecolor=BG)
    left += ct[g].values
ax.set_xlim(0, 100); ax.set_xlabel("% of books in that era"); ax.set_title("How the Genre Mix Changed Over Time")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.14), frameon=False, labelcolor=TXT)
ax.grid(axis="y", visible=False)
save(fig, "04_genre_mix_by_era.png")

# 5 — Average rating by genre (enriched only)
fig, ax = plt.subplots(figsize=(10, 5.6))
rs = E.dropna(subset=["Average_Rating"]).groupby("Genre")["Average_Rating"].agg(["mean", "count"]).sort_values("mean")
bars = ax.barh(rs.index, rs["mean"], color=BLUE)
for b, (m, n) in zip(bars, rs.values):
    ax.text(m + .03, b.get_y() + b.get_height()/2, f"{m:.2f}  (n={int(n)})", va="center", color=TXT, fontsize=10)
ax.set_xlim(0, 5.4); ax.set_xlabel("Average rating (1–5)"); ax.set_title("Average Reader Rating by Genre")
ax.grid(axis="y", visible=False)
save(fig, "05_rating_by_genre.png", "Only books with at least one rating. Small n = less reliable.")

# 6 — Top 10 most wanted books
fig, ax = plt.subplots(figsize=(11, 6))
top = E.nlargest(10, "Want_To_Read_Count").iloc[::-1]
lbl = top["Title"].str.slice(0, 38) + "  —  " + top["Primary_Author"]
ax.barh(lbl, top["Want_To_Read_Count"], color=CYAN, label="Want to read")
ax.barh(lbl, top["Already_Read_Count"], color=GREEN, label="Already read")
ax.set_title("Top 10 Most Wanted Books on Open Library"); ax.set_xlabel("Readers")
ax.legend(frameon=False, labelcolor=TXT, loc="lower right"); ax.grid(axis="y", visible=False)
save(fig, "06_top10_most_wanted.png")

# 7 — Page count distribution
fig, ax = plt.subplots(figsize=(10, 5.6))
pc = df.loc[~df["Page_Count_Imputed"], "Page_Count"].dropna()
ax.hist(pc, bins=30, color=INDIGO, edgecolor=BG)
ax.axvline(pc.median(), color=CYAN, ls="--", lw=2)
ax.text(pc.median() + 15, ax.get_ylim()[1]*.9, f"median {pc.median():.0f} pages", color=CYAN)
ax.set_title("Book Length Distribution"); ax.set_xlabel("Pages"); ax.set_ylabel("Books")
save(fig, "07_page_count_distribution.png", f"n = {len(pc)} books with a real page count (imputed values excluded)")

# 8 — Editions vs rating (popularity vs quality)
fig, ax = plt.subplots(figsize=(10, 5.8))
s = E.dropna(subset=["Average_Rating"])
for i, g in enumerate(genre_order):
    d = s[s["Genre"] == g]
    ax.scatter(d["Edition_Count"], d["Average_Rating"], s=np.clip(d["Rating_Count"], 8, 300),
               color=PALETTE[i], alpha=.75, label=g, edgecolor="none")
ax.set_xscale("log"); ax.set_xlabel("Number of editions (log scale)"); ax.set_ylabel("Average rating")
ax.set_title("Popularity (Editions) vs Reader Rating")
lg = ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.14), frameon=False, labelcolor=TXT)
for hnd in lg.legend_handles: hnd.set_sizes([50])
r = np.corrcoef(np.log10(s["Edition_Count"].clip(1)), s["Average_Rating"])[0, 1]
save(fig, "08_editions_vs_rating.png", f"Bubble size = number of ratings.  Correlation (log editions vs rating): r = {r:.2f}")

# 9 — Data-quality fixes applied (for the wrangling slide)
fig, ax = plt.subplots(figsize=(11, 5.8))
fx = pd.Series(fixes).sort_values()
bars = ax.barh(fx.index, fx.values, color=[GREEN if "imputed" in k or "standard" in k or "re-formatted" in k else PINK for k in fx.index])
ax.bar_label(bars, padding=4, color=TXT)
ax.set_xscale("log"); ax.set_xlabel("Records affected (log scale)")
ax.set_title("What the Wrangling Fixed"); ax.grid(axis="y", visible=False)
save(fig, "09_wrangling_fixes.png", "Pink = errors removed / corrected.  Green = values standardized or filled.")

# 10 — Ebook availability by genre
fig, ax = plt.subplots(figsize=(10, 5.6))
eb = df.groupby("Genre")["Ebook_Available"].mean().sort_values() * 100
bars = ax.barh(eb.index, eb.values, color=SKY)
ax.bar_label(bars, fmt="%.0f%%", padding=4, color=TXT)
ax.set_xlim(0, 100); ax.set_xlabel("% with a free full-text ebook"); ax.set_title("Ebook Availability by Genre")
ax.grid(axis="y", visible=False)
save(fig, "10_ebook_by_genre.png")

# ---------------- key insights for the report ----------------
log("\nKEY INSIGHTS")
log(f"   Largest genre: {vc.index[0]} ({vc.iloc[0]}), smallest: {vc.index[-1]} ({vc.iloc[-1]})")
log(f"   Median publish year: {int(df['Publish_Year'].median())}; busiest decade: {int(dec.idxmax())}s ({dec.max()} books)")
log(f"   Highest rated genre: {rs['mean'].idxmax()} ({rs['mean'].max():.2f}); lowest: {rs['mean'].idxmin()} ({rs['mean'].min():.2f})")
log(f"   Most wanted book: {E.nlargest(1,'Want_To_Read_Count')['Title'].iloc[0]} "
    f"({int(E['Want_To_Read_Count'].max())} want-to-read)")
log(f"   Median length: {pc.median():.0f} pages")
log(f"   Ebooks: {df['Ebook_Available'].mean():.0%} overall; highest in {eb.idxmax()} ({eb.max():.0f}%)")
log(f"   Editions vs rating correlation r = {r:.2f} (popular ≠ necessarily better rated)")

with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"\nSaved: {OUT_CSV}\nSaved: {LOG}\nCharts: {CHART_DIR}")
