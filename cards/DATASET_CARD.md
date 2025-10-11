
# Dataset Card — Rice Classification
- **Source**: (add link/citation)
- **License**: (add license)
- **Task**: Rice variety binary classification by geometric features
- **Columns**: Area, Perimiter, Major_Axis_Length, Minor_Axis_Length, Eccentricity, Convex_Area, Extent, class
- **Preprocessing**: mean-impute + MinMaxScaler; label map class1→0, class2→1
- **Split**: stratified 80/20 with random_state=0
- **Limitations**: small, mild imbalance (class2:800 / class1:600)
