# Data Cleaning

This section describes the scripts used to clean and organize the Yelp dataset for analysis. The goal was to filter the full dataset to include **only restaurant-related data** and then consolidate all relevant information into smaller, manageable, state-specific tables.

---

## 1. Filter-Script.ipynb

The **Filter-Script** processes the raw Yelp JSON files and outputs filtered CSVs containing only restaurant review-relevant data.

### Key Steps:

1. **Filter businesses**:  
   - Loads `business.json` and keeps only businesses in restaurant-related categories (`Restaurants`, `Food`, etc.).
   - Saves the filtered businesses as `business.csv`.

2. **Filter reviews**:  
   - Loads `review.json` and keeps only reviews for the filtered restaurants.
   - Utilizes chunked processing for time and memory efficiency.
   - Saves the filtered reviews as `review.csv`.

3. **Filter other datasets**:  
   - `checkin.json` and `tip.json` are filtered to include only entries for the filtered restaurants.
   - `user.json` is filtered to include only users who wrote reviews in the filtered review dataset.

4. **Output**:  
   - All filtered datasets are saved as CSV for easier analysis, while preserving all relevant data.

---

## 2. Association-Script.ipynb

The **Association-Script** merges all filtered datasets and produces **state-specific DataFrames** for easier handling of large datasets.

### Key Steps:

1. **Rename columns** to indicate their source:  
   - `review_` → review data  
   - `business_` → business data  
   - `checkin_` → checkin summary  
   - `tip_` → tip summary (`num_tips` only, tip content removed)  
   - `user_` → user data  

2. **Aggregate multi-row datasets**:  
   - Checkins are aggregated into a `total_checkins` count per business.  
   - Tips are aggregated into `num_tips` per business.

3. **Merge datasets**:  
   - Reviews are merged with business info using `business_id`.  
   - Checkins and tips are merged at the business level.  
   - User info is merged using `user_id`.  

4. **Split by state**:  
   - The consolidated DataFrame is split into smaller DataFrames for each `business_state`.  
   - Each state-specific DataFrame is saved separately (CSV) for memory-efficient analysis.

5. **Output**:  
   - Each row contains all relevant review, business, checkin, tip, and user information for a single review.  
   - The resulting files are ready for analysis and modeling.

> **Note:** Column prefixes make it easy to distinguish the source of each field. Splitting by state avoids memory issues caused by a single massive DataFrame.

---

### Summary

Together, these two scripts:

- Reduce the raw Yelp dataset to only **restaurant-relevant data**.  
- Produce **clean, structured CSVs** for analysis.  
- Provide **state-specific consolidated DataFrames** linking reviews, users, businesses, checkins, and tips.  
- Make working with very large datasets more manageable and memory-efficient.