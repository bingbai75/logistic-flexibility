# Replication Data and Code for "The Value of Logistic Flexibility in E-commerce"

## Prerequisite 
Install Anaconda and create environment as follows:
```bash
conda create -n logistic_flexibility python==3.11.5 -y
conda activate logistic_flexibility
pip install -U pip
pip install -r requirements.txt
```

Then run
```bash
conda activate logistic_flexibility && python Bai2025-Structural.py
```
**Note**: The runtime may exceed 2 hours depending on GPU specifications.

The materials to replicate the results in the paper include:

## 1. SYNTHETIC data: 
### Reduced form evidence: `Bai2025_agg_synthetic_data.rda`.  
### Structural estimation & counterfactual analyses: 
a. `customer_data.pickle`
b. `customer_all.csv`
c. `customer_info_loc.csv`
d. `gaussian_params.csv`
e. `prob_slot.csv`
f. `station_coordinates_perturbed.csv`
The actual dataset used in the study is protected under NDA and cannot be shared publicly. To facilitate replication, we provide a synthetic dataset designed to mimic key statistical properties of the original data. The provided replication code will reproduce all main tables in the paper, but the exact numerical outputs may differ due to the use of synthetic data.


## 2. Replication Code 
* Reduced form evidence: `Bai2025-Empirical Evidence.Rmd`
* Structural estimation & counterfactual analyses: `Bai2025-Structural.py`

## 3. Data Dictionary:
### `Bai2025_agg_data.rda`

Aggregated location-level data for reduced form evidence analysis, including simulated sales metrics, order statistics, and treatment-related variables.

**Format**: R data file (.rda)

**Columns**:
- Column 1: `location_id`, location i
- Column 2: `year_week`, year-week indicator
- Column 3: `week_day`, weekday indicator
- Column 4: `Days`, the number of days after the treatment 
- Column 5: `distance`, the distance between location i and the nearest newly opened station
- Column 6: `town_id`, area indicator
- Column 7: `GMV`, the gross merchandise volume, i.e. total sales at location i on day t
- Column 8: `Items_value`, the average items value per order at location i on day t
- Column 9: `Orders`, number of orders at location i on day t
- Column 10: `Newcomers`, number of new consumers at location i on day t

---

### `customer_data.pickle`

Customer behavior data containing simulated delivery patterns and station usage information.

**Format**: Pickle file (pandas DataFrame)

**Columns**:
- Column 1: `person_id`, unique customer identifier 
- Column 2: `n_station`, the number of station deliveries 
- Column 3: `n_home`, the number of home deliveries 
- Column 4: `closest_distance`, the distance between customer and the nearest station 
- Column 5: `pickup`, NumPy array of pickup time patterns (shape: n_station × 42, where n_station is the number of station deliveries for this customer; each row has one '1' indicating the pickup time slot, remaining values are '0'; NaN if n_station = 0)
- Column 6: `deliver`, NumPy array of delivery time patterns (shape: n_station × 42, where n_station is the number of station deliveries for this customer; each row has consecutive '1's from a starting position to column 41, indicating delivery time window; NaN if n_station = 0)

**Note**: Only customers with `n_station > 0` have non-NaN `pickup` and `deliver` arrays. The array dimensions vary by customer based on their `n_station` value.

---

### `customer_all.csv`

Customer basic information including simulated order amounts and item statistics.

**Format**: CSV file (comma-separated)

**Columns**:
- Column 1: `person_id`, unique customer identifier
- Column 2: `month_order_amount`, the total monthly order amount for the customer 
- Column 3: `mean_item_quantity`, the average number of items per order for the customer 
- Column 4: `mean_max_item_pay`, the average maximum item payment per order for the customer 
- Column 5: `mean_avg_item_pay`, the average item payment per order for the customer 

---

### `customer_info_loc.csv`

Customer location-based information with category preferences and order statistics (across the whole city).

**Format**: CSV file (comma-separated)

**Columns**:
- Column 1: `person_id`, unique customer identifier (format: `lng_grid_lat_grid_id`, e.g., `960_625_1`)
- Column 2: `closest_distance`, the distance between customer and the nearest station 
- Column 3: `month_order_amount`, the total monthly order amount for the customer 
- Column 4: `mean_item_quantity`, the average number of items per order for the customer
- Column 5: `mean_max_item_pay`, the average maximum item payment per order for the customer
- Column 6: `mean_avg_item_pay`, the average item payment per order for the customer 


---

### `gaussian_params.csv`

Gaussian mixture model parameters for delivery time window distributions.

**Format**: CSV file (comma-separated)

**Columns**:
- Column 1: `parameter`, parameter name (string: `mu1`, `mu2`, `sigma_est1`, `sigma_est2`, `p_dis1`, `p_dis2`)
- Column 2: `value`, parameter value (float)

**Parameter Descriptions**:
- `mu1`: mean of the first Gaussian component (morning delivery window)
- `mu2`: mean of the second Gaussian component (afternoon delivery window)
- `sigma_est1`: standard deviation of the first Gaussian component
- `sigma_est2`: standard deviation of the second Gaussian component
- `p_dis1`: probability weight of the first Gaussian component (morning slot)
- `p_dis2`: probability weight of the second Gaussian component (afternoon slot)

---

### `prob_slot.csv`

Time slot probability distributions for home and station delivery patterns.

**Format**: CSV file (comma-separated)

**Columns**:
- Column 1: `prob_slot_h`, probability of home delivery occurring in each of 14 time slots 
- Column 2: `prob_slot_s`, probability of station delivery occurring in each of 14 time slots 

**Note**: Each column contains 14 values representing probabilities for different time slots. The 14 time slots correspond to specific operation hours of the day.

---

### `station_coordinates_perturbed.csv`

Perturbed station coordinates in grid space.

**Format**: CSV file (comma-separated)

**Columns**:
- Column 1: `station_id`, unique station identifier
- Column 2: `lng_grid`, longitude grid coordinate of the station
- Column 3: `lat_grid`, latitude grid coordinate of the station 

**Note**: The coordinates are in grid units (not actual geographic coordinates) and have been perturbed from original values for privacy protection.


	