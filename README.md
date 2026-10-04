# Data-Driven Book Writing: Identifying Promising Genres for the Future of Books

## Project Overview

This project uses data wrangling and analysis to identify promising book genres for future book writing. The project focuses on collecting book metadata, cleaning and enriching the dataset, and analyzing reader-interest patterns.

## Objectives

- Collect book data using the Open Library API.
- Clean and prepare the collected dataset.
- Enrich the dataset with additional book information.
- Analyze genre distribution, ratings, reader-interest indicators, and ebook availability.
- Identify promising genres for future book writing.

## Data Collection

The raw dataset was collected from the Open Library API.

- Raw records: 2,081
- Raw columns: 17
- Cleaned records: 2,075
- Analytical fields after processing: 33
- Enriched records: 419 (approximately 20%)

## Data Wrangling

The main data-cleaning and wrangling steps included:

- Removing duplicate records
- Handling missing page-count values
- Standardizing and preparing genre information
- Enriching book records with additional metadata
- Preparing the final dataset for analysis and visualization

## Key Findings

- Romance was the largest genre in the final dataset, with 281 books.
- Young Adult had the highest mean rating (approximately 4.36), but this result was based on only 10 enriched rating records and should therefore be interpreted cautiously.
- Eight of the top ten most-wanted books in the enriched subset were classified as Romance.
- Thriller showed meaningful reader-interest activity and very high ebook availability.
- The correlation between edition count and average rating was weak (r = 0.09), indicating that edition count alone is not a strong predictor of book success.

## Recommendation

Based on the available evidence, Romance is the strongest choice for a new book when the primary objective is reader interest.

Young Adult is a promising alternative because of its high average rating, but more data is needed for a reliable conclusion. Thriller is also a potential secondary option.

These recommendations indicate observed reader-interest patterns and do not guarantee commercial success.

## Limitations

- Sales data was not available, so no sales-based conclusion was made.
- Open Library data does not represent the complete publishing market.
- Genre classification depends on Open Library metadata.
- Some enriched analyses were based on a limited subset of records.

## Conclusion

The project demonstrates how data wrangling can transform the question of which genre to choose for a future book into a structured, data-driven decision problem. The analysis suggests Romance as the strongest evidence-based opportunity based on observed reader-interest patterns, while Young Adult and Thriller remain promising alternatives.

