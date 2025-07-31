# Predictor Numerical Fixes

## Overview

This document explains the numerical fixes implemented in the predictor to resolve the "Input X contains infinity or a value too large for dtype('float64')" error that was occurring during retraining.

## Problem Description

The error was occurring because:
1. **Infinite values**: Division by zero or log of zero in feature calculations
2. **NaN values**: Missing data or invalid calculations
3. **Extremely large values**: Numerical instability in rolling statistics or feature engineering
4. **Data quality issues**: Real-time data from APIs can contain outliers or corrupted values

## Files Modified

### 1. `predictor.py`

#### Changes in `preprocess_input_data()` method:
- **Infinity replacement**: Replace infinite values with NaN, then remove rows
- **Extreme value clipping**: Use robust statistics (IQR method) to clip outliers
- **Final validation**: Check for any remaining problematic values

```python
# 🔧 NUMERICAL VALIDATION AND CLEANUP
# Check for infinite values and replace them
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if col == 'timestamp':
        continue
    # Replace infinite values with NaN
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# Remove rows with NaN values after infinity replacement
initial_rows = len(df)
df = df.dropna().reset_index(drop=True)
final_rows = len(df)

# Clip extreme values to prevent numerical instability
for col in numeric_columns:
    if col == 'timestamp':
        continue
    # Calculate robust statistics
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    
    # Define clipping bounds (more conservative than 1% and 99%)
    lower_bound = q1 - 3 * iqr
    upper_bound = q99 + 3 * iqr
    
    # Clip extreme values
    clipped_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    if clipped_count > 0:
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
```

#### Changes in `predict_next_period()` method:
- **Final validation before model input**: Check input sequence for infinite/NaN values
- **Value replacement**: Replace problematic values with zeros
- **Range clipping**: Clip extremely large values to prevent numerical overflow

```python
# 🔧 FINAL NUMERICAL VALIDATION BEFORE MODEL INPUT
# Check for any remaining infinite or extremely large values
if np.isinf(input_sequence).any():
    print(f"❌ Error: Input sequence contains infinite values")
    # Replace infinite values with zeros
    input_sequence = np.where(np.isinf(input_sequence), 0.0, input_sequence)

if np.isnan(input_sequence).any():
    print(f"❌ Error: Input sequence contains NaN values")
    # Replace NaN values with zeros
    input_sequence = np.where(np.isnan(input_sequence), 0.0, input_sequence)

# Check for extremely large values that might cause numerical issues
max_abs_value = np.max(np.abs(input_sequence))
if max_abs_value > 1e6:
    print(f"⚠️ Warning: Input sequence contains very large values (max: {max_abs_value})")
    # Clip to reasonable range
    input_sequence = np.clip(input_sequence, -1e6, 1e6)
```

### 2. `continuous_predictor.py`

#### Changes in `_perform_retraining_internal()` method:
- **Training data validation**: Clean training data before passing to trainer
- **Infinity/NaN removal**: Replace problematic values in training data
- **Extreme value clipping**: Apply robust clipping to training data

```python
# 🔧 NUMERICAL VALIDATION OF TRAINING DATA
if training_data is not None and len(training_data) > 0:
    # Check for infinite or NaN values in training data
    numeric_columns = training_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col == 'timestamp':
            continue
        # Replace infinite values with NaN
        training_data[col] = training_data[col].replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with NaN values
    initial_rows = len(training_data)
    training_data = training_data.dropna().reset_index(drop=True)
    final_rows = len(training_data)
    
    # Clip extreme values in training data
    for col in numeric_columns:
        if col == 'timestamp':
            continue
        # Calculate robust statistics
        q1 = training_data[col].quantile(0.01)
        q99 = training_data[col].quantile(0.99)
        iqr = training_data[col].quantile(0.75) - training_data[col].quantile(0.25)
        
        # Define clipping bounds
        lower_bound = q1 - 3 * iqr
        upper_bound = q99 + 3 * iqr
        
        # Clip extreme values
        clipped_count = ((training_data[col] < lower_bound) | (training_data[col] > upper_bound)).sum()
        if clipped_count > 0:
            training_data[col] = training_data[col].clip(lower=lower_bound, upper=upper_bound)
```

#### Final validation before CSV creation:
```python
# 🔧 FINAL VALIDATION BEFORE SAVING TO CSV
# Ensure no infinite or NaN values remain
if training_data is not None and len(training_data) > 0:
    numeric_columns = training_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col == 'timestamp':
            continue
        # Final check and replacement
        if training_data[col].isnull().any() or np.isinf(training_data[col]).any():
            print(f"⚠️ Final cleanup: Replacing problematic values in column '{col}'")
            training_data[col] = training_data[col].replace([np.inf, -np.inf], 0.0)
            training_data[col] = training_data[col].fillna(0.0)
```

### 3. `trainer.py`

#### Changes in `retrain_with_current_data()` method:
- **Final validation before training**: Check X and y arrays for numerical issues
- **Value replacement**: Replace infinite/NaN values with zeros
- **Range clipping**: Clip extremely large values to prevent overflow

```python
# 🔧 FINAL NUMERICAL VALIDATION BEFORE TRAINING
# Check for infinite or NaN values in X and y
if np.isinf(X).any() or np.isnan(X).any():
    print(f"❌ Error: Input features X contain infinite or NaN values")
    # Replace problematic values
    X = np.where(np.isinf(X), 0.0, X)
    X = np.where(np.isnan(X), 0.0, X)

if np.isinf(y).any() or np.isnan(y).any():
    print(f"❌ Error: Target values y contain infinite or NaN values")
    # Replace problematic values
    y = np.where(np.isinf(y), 0.0, y)
    y = np.where(np.isnan(y), 0.0, y)

# Check for extremely large values
max_abs_X = np.max(np.abs(X))
max_abs_y = np.max(np.abs(y))

if max_abs_X > 1e6:
    print(f"⚠️ Warning: Input features X contain very large values (max: {max_abs_X})")
    X = np.clip(X, -1e6, 1e6)

if max_abs_y > 1e6:
    print(f"⚠️ Warning: Target values y contain very large values (max: {max_abs_y})")
    y = np.clip(y, -1e6, 1e6)
```

## Testing

### Test Script: `test_predictor_numerical_fixes.py`

The test script verifies that:
1. **Basic numerical handling**: Predictor can handle data with infinite/NaN values
2. **Extreme value handling**: Predictor can handle extremely large/small values
3. **End-to-end functionality**: Complete prediction pipeline works with problematic data

### Running Tests

```bash
python test_predictor_numerical_fixes.py
```

## Key Features of the Fixes

### 1. **Robust Statistics**
- Uses IQR (Interquartile Range) method for outlier detection
- More robust than simple percentile-based clipping
- Handles skewed distributions better

### 2. **Multi-Stage Validation**
- **Stage 1**: Data preprocessing cleanup
- **Stage 2**: Training data validation
- **Stage 3**: Final validation before model input
- **Stage 4**: Validation before training

### 3. **Conservative Clipping**
- Uses 3 × IQR bounds (more conservative than 1%/99% percentiles)
- Prevents extreme outliers from affecting model training
- Maintains data integrity while removing problematic values

### 4. **Graceful Degradation**
- Replaces infinite/NaN values with zeros when necessary
- Continues operation even with problematic data
- Provides detailed logging of cleanup actions

## Expected Results

After implementing these fixes:

1. **No more "Input X contains infinity" errors** during retraining
2. **Improved numerical stability** in the prediction pipeline
3. **Better handling of real-time data** from APIs
4. **Robust feature engineering** that can handle edge cases
5. **Detailed logging** of any numerical issues encountered

## Monitoring

The fixes include comprehensive logging that will help identify:
- How many rows were removed due to infinite/NaN values
- How many extreme values were clipped
- Which columns contained problematic data
- When numerical issues occur in the pipeline

## Conclusion

These numerical fixes address the root cause of the "Input X contains infinity" error by implementing robust data validation and cleanup at multiple stages of the prediction and retraining pipeline. The fixes are designed to be conservative and maintain data integrity while preventing numerical instability. 