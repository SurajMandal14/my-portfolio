# 🎯 Complete Application-as-Recipe Guide for Fuzzy Matching

## Following Your Senior's Instructions

---

## 📋 Overview of What We're Building

```
Dataiku Project Structure:
├── Zone: FUZZY_MATCHING_APP
    ├── 📁 input1_folder (Managed Folder)
    ├── 📁 input2_folder (Managed Folder)
    ├── 🐍 fuzzy_matching_recipe (Python Recipe)
    ├── 📁 output_folder (Managed Folder)
    └── ⚙️ build_scenario (Scenario)
```

**User Experience**:

1. User selects recipe from menu
2. Connects their datasets
3. Enters ID columns and comparison columns
4. Runs recipe → Gets matched results

---

## 🚀 STEP 1: Create New Dataiku Project

### 1.1: Create Project

1. **Go to Dataiku homepage**
2. **Click "+ NEW PROJECT"** (top-right)
3. **Fill in**:
   ```
   Project Name: Fuzzy Matching Application
   Project Key: FUZZY_APP
   ```
4. **Click "CREATE"**

**Why**: This project will become your reusable recipe.

---

## 📂 STEP 2: Create Folder Structure (Zone)

### 2.1: Create Input Folder 1

1. **In your project, click "+ DATASET"** dropdown → **"Folder"**

2. **Select "Create Folder"**

3. **Configure**:

   ```
   Folder name: input1_folder
   Connection: Filesystem (default)
   Path: /fuzzy_app/input1
   ```

4. **Click "CREATE"**

**Visual in Flow**: You'll see a folder icon labeled `input1_folder`

---

### 2.2: Create Input Folder 2

Repeat same steps:

```
Folder name: input2_folder
Connection: Filesystem
Path: /fuzzy_app/input2
```

---

### 2.3: Create Output Folder

```
Folder name: output_folder
Connection: Filesystem
Path: /fuzzy_app/output
```

**Your Flow now shows**:

```
[input1_folder]    [input2_folder]    [output_folder]
```

---

## 🐍 STEP 3: Create Python Recipe

### 3.1: Create Recipe

1. **Click on `input1_folder`** in the Flow (left-click to select)

2. **Right panel** → **"Actions"** → **"+ Recipe"**

3. **Choose "Python"** recipe

4. **Recipe dialog opens**:

   ```
   Input folders:
   - input1_folder ✓ (already selected)
   + Add another input → Select: input2_folder

   Output folder:
   + Add output → Select: output_folder

   Recipe name: fuzzy_matching_recipe
   ```

5. **Click "CREATE RECIPE"**

---

### 3.2: Understand Default Code & Recipe API

The Python editor opens with default Dataiku code. **Instead of using folder IDs, we'll use the `recipe` API** (same as your manager's EDA tool):

```python
# OLD WAY (hardcoded folder IDs):
# input1_folder = dataiku.Folder("3vUV3r4h")

# NEW WAY (dynamic recipe API - RECOMMENDED):
from dataiku import recipe
input_folder_1 = recipe.get_input(0)  # First input
input_folder_2 = recipe.get_input(1)  # Second input
output_folder = recipe.get_output(0)  # Output
```

**Why this is better**:

- ✅ No hardcoded IDs or names
- ✅ Works automatically with whatever folders user connects
- ✅ Same pattern as your manager's EDA tool
- ✅ Proper Dataiku recipe best practice

---

### 3.3: Replace with Full Recipe Code

**Delete all default code** and **replace with this complete implementation**:

```python
"""
Fuzzy Matching Recipe
=====================
Reads Excel files from two input folders, performs fuzzy matching,
and writes results to output folder.
"""

import dataiku
from dataiku import recipe
import pandas as pd
import numpy as np
import re
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
import warnings
import os
from io import BytesIO

warnings.filterwarnings("ignore")

# ====================
# GET CUSTOM VARIABLES (User inputs or defaults)
# ====================
vars = dataiku.get_custom_variables()

# Get user-defined columns or use defaults
ID_COLUMN_1 = vars.get('id_column_1', 'id')
COMPARISON_COLUMN_1 = vars.get('comparison_column_1', 'company_name')
ID_COLUMN_2 = vars.get('id_column_2', 'id')
COMPARISON_COLUMN_2 = vars.get('comparison_column_2', 'company_name')
THRESHOLD = int(vars.get('threshold', 80))
LOGIC_TYPE = vars.get('logic_type', 'Fuzzy Matching')

print(f"Configuration:")
print(f"  Logic: {LOGIC_TYPE}")
print(f"  File 1 - ID: {ID_COLUMN_1}, Comparison: {COMPARISON_COLUMN_1}")
print(f"  File 2 - ID: {ID_COLUMN_2}, Comparison: {COMPARISON_COLUMN_2}")
print(f"  Threshold: {THRESHOLD}%")

# ====================
# UTILITY FUNCTIONS
# ====================

def cleaning_string(string):
    """Clean and normalize text for fuzzy matching."""
    string = str(string)
    string = fix_text(string)
    string = string.encode("ascii", errors="ignore").decode()
    string = string.lower()
    string = re.sub(r" +", r" ", string).strip()
    string = re.sub(r"\.co[a-z+?\.?[a-z]+", r"", string)
    string = re.sub(r"^\s+| \s+?", r"", string)
    string = re.sub(r"[^\w\s]", r" ", string)

    # Remove country names
    key = r"""(?<!^)(\b(north latam|european|middle east|europe|asia pacific|asia|myanmar|nederland|albania|algeria|angola|antigua and barbuda|argentina|armenia|aruba|australia|austria|azerbaijan|bahamas|bahrain|bangladesh|barbados|belarus|belgium|belize|benin|bermuda|bolivia|botswana|bouvet island|brazil|brunei darussalam|bulgaria|burkina faso|cambodia|cameroon|canada|cayman islands|chile|china|christmas island|colombia|costa rica|cote d'ivoire|croatia|cyprus|czech republic|denmark|dominica|dominican republic|ecuador|egypt|el salvador|eritrea|estonia|fiji|finland|france|french polynesia|gabon|gambia|georgia|germany|ghana|gibraltar|greece|guatemala|guernsey|holy see|honduras|hong kong|hungary|iceland|india|indonesia|iran|iraq|ireland|channel islands|israel|italy|jamaica|japan|jersey|jordan|kazakhstan|kenya|kiribati|korea|south korea|kuwait|latvia|lebanon|liberia|libyan arab jamahiriya|liechtenstein|lithuania|luxembourg|macau|madagascar|malaysia|maldives|mali|malta|marshall islands|mauritius|mexico|monaco|mongolia|montenegro|morocco|mozambique|nepal|netherlands|netherlands antilles|new caledonia|new zealand|niger|nigeria|norway|oman|pakistan|palestinian territory, occupied|panama|papua new guinea|paraguay|peru|philippines|poland|portugal|puerto rico|qatar|romania|russian federation|russia|rwanda|saint kitts and nevis|samoa|saudi arabia|senegal|serbia|seychelles|sierra leone|singapore|slovakia|slovenia|solomon islands|south africa|spain|sri lanka|sudan|suriname|swaziland|sweden|switzerland|taiwan|tanzania|thailand|togo|trinidad and tobago|tunisia|turkey|turks and caicos islands|uganda|ukraine|united arab emirates|uae|united kingdom|united states|usa|uruguay|brasil|uzbekistan|venezuela|vietnam|virgin islands, british|british virgin islands|virgin islands, u\.s\.|yemen|zambia|moldova|nicaragua|bosnia|vatican city state|london)\b)"""
    string = re.sub(key, r"", string)
    string = string.upper()

    # Remove company suffixes
    key1 = r"""((?<!^)(\b(REAL ESTATE|4P|ASS|ASSOCIATES|ASSOCIATION|BHD|BV|CG|CMB|CO|COLTD|COMMERCIAL|COMP|COMPANY|CORP|CORPORATION|CRP|CV|DE CV|DMCC|EL|ENTERPRISE|ENTERPRISES|EST|ESTATE|ET|EXP|FUND|FZ|FZE|GB|GMBH|GROA|GROUP|GRP|GRUPO|HOLDING|HOLDINGS|IMP|INC|INCORPORATED|IND|INDUST|INDUSTRIES|INDUSTRY|INT|INTERN|INTERNATIONAL|INTL|INV|JOINT STOCK|JSC|KFT|L L C|L LTD|LC|LIM|LIMI|LIMIT|LIMITE|LIMITED|LLC|LLP|LP|LT|LTD|MA|MGT|MME|MR|MRS|NBFI|NV|OYJ|PARTNERS|PJSC|PL|PLC|PRIVATE|PRODUCTS|PTE|PTY|PUBLIC|PVT|RE|S A|S R O|SA|SA DE CV|SAB|SAE|SAL|SARL|SDN|SDV|SE|SERVICE|SERVICES|SME|SOLUTION|SOLUTIONS|SON|SONS|SPA|SRL|SRO|ST|TBK|TRAD|TRADING|TRD|TRDG|TRUST|TRUSTEE|TST|UNASSIGNED|VE|WPP|C\\O|TD|WLL|W L L|CO.|SAS|LTDA|LTRD|AG|KG|DOO|DEPT|SP Z O O)\b)).*$"""
    string = re.sub(key1, r"", string)
    string = string.lower()
    string = re.sub("(BRANCH)", r"", string)
    string = string.replace("  ", " ")

    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = "[" + re.escape("".join(chars_to_remove)) + "]"
    string = re.sub(rx, "", string)

    string = string.replace("&", "and")
    string = string.replace(",", " ")
    string = string.replace("-", " ")
    string = string.title()
    string = re.sub(" +", " ", string).strip()
    string = " " + string + " "
    string = re.sub(r"[,-./]|\sBD", r"", string)
    string = string.replace("  ", " ")
    return string


def perform_fuzzy_matching(df1, df2, id_col1, comp_col1, id_col2, comp_col2, threshold_percent, is_dedup=False):
    """Perform fuzzy matching between two dataframes."""
    try:
        df1_copy = df1.copy()
        df2_copy = df2.copy()

        # Clean the comparison columns
        df1_copy['cleaned'] = df1_copy[comp_col1].apply(cleaning_string)
        df2_copy['cleaned'] = df2_copy[comp_col2].apply(cleaning_string)

        df1_copy.dropna(subset=['cleaned'], inplace=True)
        df2_copy.dropna(subset=['cleaned'], inplace=True)

        df1_copy.reset_index(drop=True, inplace=True)
        df2_copy.reset_index(drop=True, inplace=True)

        if df1_copy.empty or df2_copy.empty:
            return pd.DataFrame()

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', ngram_range=(2, 4))
        all_cleaned_text = pd.concat([df1_copy['cleaned'], df2_copy['cleaned']]).unique()
        vectorizer.fit(all_cleaned_text)

        tfidf_matrix1 = vectorizer.transform(df1_copy['cleaned'])
        tfidf_matrix2 = vectorizer.transform(df2_copy['cleaned'])

        matches = sp_matmul_topn(tfidf_matrix1, tfidf_matrix2.T, top_n=10, threshold=threshold_percent/100.0)

        def get_matches_df(matches_matrix, A, B, A_id_col, B_id_col, A_comp_col, B_comp_col):
            rows = matches_matrix.tocoo()
            data = []
            for row, col, score in zip(rows.row, rows.col, rows.data):
                if is_dedup and A.loc[row, A_id_col] == B.loc[col, B_id_col]:
                    continue

                id1 = A.loc[row, A_id_col]
                comp1 = A.loc[row, A_comp_col]
                id2 = B.loc[col, B_id_col]
                comp2 = B.loc[col, B_comp_col]
                data.append([id1, comp1, id2, comp2, round(score * 100, 2)])

            if is_dedup:
                seen = set()
                unique_data = []
                for item in data:
                    key = tuple(sorted((item[0], item[2])))
                    if key not in seen:
                        unique_data.append(item)
                        seen.add(key)
                data = unique_data

            return pd.DataFrame(data, columns=['ID_File1', 'Comp_Value_File1', 'ID_File2', 'Comp_Value_File2', 'Similarity'])

        result_df = get_matches_df(matches, df1_copy, df2_copy, id_col1, id_col2, comp_col1, comp_col2)
        return result_df

    except Exception as e:
        raise Exception(f"Error in fuzzy matching: {str(e)}")


def read_file_from_folder(folder):
    """Read single file from managed folder (CSV or Excel)."""
    paths = folder.list_paths_in_partition()
    if not paths:
        raise ValueError("Input folder is empty. Please upload a file.")

    # Read first file found
    file_path = paths[0]
    file_name = os.path.basename(file_path)

    print(f"Reading: {file_name}")

    with folder.get_download_stream(file_path) as f:
        if file_name.lower().endswith(".csv"):
            return pd.read_csv(f), file_name
        elif file_name.lower().endswith((".xlsx", ".xls")):
            buffer = BytesIO(f.read())
            if file_name.lower().endswith(".xlsx"):
                return pd.read_excel(buffer, engine="openpyxl"), file_name
            else:
                return pd.read_excel(buffer, engine="xlrd"), file_name
        else:
            raise ValueError(f"Unsupported file type: {file_name}. Use CSV or Excel files.")


# ====================
# READ INPUT FOLDERS USING RECIPE API
# ====================
print("Reading input folders...")

# Get inputs dynamically from recipe
input_folder_1 = recipe.get_input(0)
input_folder_2 = recipe.get_input(1)
output_folder = recipe.get_output(0)

# Read files
df1, file1_name = read_file_from_folder(input_folder_1)
df2, file2_name = read_file_from_folder(input_folder_2)

print(f"Dataset 1 ({file1_name}): {df1.shape}")
print(f"Dataset 2 ({file2_name}): {df2.shape}")

# For deduplication, use same file for both
if LOGIC_TYPE == 'De-duplication':
    df2 = df1.copy()
    file2_name = file1_name
    ID_COLUMN_2 = ID_COLUMN_1
    COMPARISON_COLUMN_2 = COMPARISON_COLUMN_1
    print("Mode: De-duplication (using file 1 for both sides)")

# ====================
# VALIDATE COLUMNS
# ====================
if ID_COLUMN_1 not in df1.columns:
    raise ValueError(f"Column '{ID_COLUMN_1}' not found in file 1. Available: {list(df1.columns)}")
if COMPARISON_COLUMN_1 not in df1.columns:
    raise ValueError(f"Column '{COMPARISON_COLUMN_1}' not found in file 1. Available: {list(df1.columns)}")
if ID_COLUMN_2 not in df2.columns:
    raise ValueError(f"Column '{ID_COLUMN_2}' not found in file 2. Available: {list(df2.columns)}")
if COMPARISON_COLUMN_2 not in df2.columns:
    raise ValueError(f"Column '{COMPARISON_COLUMN_2}' not found in file 2. Available: {list(df2.columns)}")

# ====================
# PERFORM FUZZY MATCHING
# ====================
print("Performing fuzzy matching...")

is_dedup = (LOGIC_TYPE == 'De-duplication')
result_df = perform_fuzzy_matching(
    df1, df2,
    ID_COLUMN_1, COMPARISON_COLUMN_1,
    ID_COLUMN_2, COMPARISON_COLUMN_2,
    THRESHOLD,
    is_dedup=is_dedup
)

if is_dedup:
    result_df.columns = ['Record_ID_1', 'Original_Value', 'Record_ID_2_Duplicate', 'Duplicate_Value', 'Similarity_Percent']
else:
    result_df.columns = ['ID_File1', 'Value_File1', 'ID_File2', 'Value_File2', 'Similarity_Percent']

print(f"Found {len(result_df)} matches")

# ====================
# WRITE OUTPUT
# ====================
print("Writing output...")

output_folder = dataiku.Folder("output_folder")

# Write as Excel
# Generate output filename with timestamp
timestamp = pd.Timestamp.now().strftime('%m%d%Y_%H%M')
output_filename = f"fuzzy_results_{timestamp}.xlsx"

# Write Excel file to output folder
buffer = BytesIO()
result_df.to_excel(buffer, index=False, engine='openpyxl')
buffer.seek(0)
output_folder.upload_stream(output_filename, buffer)

print(f"✓ Results written to: {output_filename}")
print(f"✓ Found {len(result_df)} matches")
print("✓

### 3.4: Important Code Notes

#### **Folder Names vs Folder IDs**
Key Differences from Old Approach

**What Changed**:

| Old Approach | New Approach (Recipe API) |
|-------------|---------------------------|
| `dataiku.Folder("folder_id")` | `recipe.get_input(0)` |
| `client.get_project().get_variables()` | `dataiku.get_custom_variables()` |
| `output_folder.get_writer()` | `output_folder.upload_stream()` |
| Hardcoded folder references | Dynamic recipe connections |

**Benefits**:
- Same pattern as your manager's EDA tool
- No folder IDs or names to worry about
- Works with any folders user connects
- Cleaner, more maintainable code
### 3.5: Save and Test Recipe

### 3.5: Save and Test Recipe

1. **Click "SAVE"** (top-left)
2. **Upload sample files** to `input1_folder` and `input2_folder`:
   - Go to Flow → Click folder → "Upload files" → Select Excel/CSV
3. **Click "RUN"** to test it (may fail initially if variables not set - that's okay!)

**Expected behavior**:

- ✅ If variables are set: Recipe runs, output created
- ⚠️ If variables not set: Error about missing columns (we'll fix with defaults next)

---

## 🔧 STEP 4: Set Up Project Variables (User Inputs)

### 4.1: Go to Project Variables

1. **Click your project name** (top-left) → **"Settings"**

2. **Left sidebar** → **"Variables"**

3. **You'll see sections**: Local Variables, Project Variables, etc.

---

### 4.2: Add Variables

Click **"+ ADD"** for each variable:

#### Variable 1: Logic Type

```

Name: logic_type
Type: String
Default Value: Fuzzy Matching
Description: Type of matching (Fuzzy Matching or De-duplication)

```

#### Variable 2: ID Column 1

```

Name: id_column_1
Type: String
Default Value: id
Description: ID column name in first file (default: id)

```

#### Variable 3: Comparison Column 1

```

Name: comparison_column_1
Type: String
Default Value: company_name
Description: Column to compare in first file (default: company_name)

```

#### Variable 4: ID Column 2

```

Name: id_column_2
Type: String
Default Value: id
Description: ID column name in second file (default: id)

```

#### Variable 5: Comparison Column 2

```

Name: comparison_column_2
Type: String
Default Value: company_name
Description: Column to compare in second file (default: company_name)

```

#### Variable 6: Threshold

```

Name: threshold
Type: String (or Integer)
Default Value: 80
Description: Similarity threshold percentage (0-100)

```

**Why**: These variables act as user inputs when the recipe is instantiated.

**Default values ensure**: If user doesn't provide values, recipe won't fail.

---

## ⚙️ STEP 5: Create Scenario (Build Step)

### 5.1: Create Scenario

1. **Top navigation** → **"Scenarios"** tab

2. **Click "+ NEW SCENARIO"**

3. **Name it**: `build_fuzzy_results`

4. **Click "CREATE"**

---

### 5.2: Add Build Step

1. **Scenario editor opens**

2. **Click "+ ADD STEP"**

3. **Choose "Build / Train"**

4. **Configure**:

```

What to build: output_folder
Build mode: Force rebuild

```

5. **Click "SAVE"**

---

### 5.3: Add Run Recipe Step (Alternative)

Or you can directly run the recipe:

1. **"+ ADD STEP"**

2. **Choose "Run Recipe"**

3. **Select**: `fuzzy_matching_recipe`

4. **Click "SAVE"**

---

### 5.4: Activate Scenario

1. **Toggle switch at top**: "Active" → **ON**

**Why**: When users run the recipe, this scenario executes automatically.

---

## 🎨 STEP 6: Convert to Application-as-Recipe

### 6.1: Open Application Designer

1. **Click your project name** (top-left)

2. **Dropdown menu** → **"More options"** → **"Application Designer"**

(Or might be under **"Settings"** → **"Application Designer"**)

3. **Choose**: "Convert to Application-as-Recipe"

4. **Application Designer opens**

---

### 6.2: Configure Application Header

```

┌─────────────────────────────────────┐
│ Application Header │
├─────────────────────────────────────┤
│ Recipe Name: │
│ [Fuzzy Matching Tool] │
│ │
│ Description: │
│ [Intelligent fuzzy matching and...] │
│ │
│ Permission: │
│ [✓] All users can instantiate │
└─────────────────────────────────────┘

```

---

### 6.3: Configure Recipe Definition

```

┌─────────────────────────────────────┐
│ Recipe Definition │
├─────────────────────────────────────┤
│ Icon: icon-search │
│ Category: Data Quality │
│ │
│ Inputs: │
│ [+ Add Input] │
│ Name: input1 │
│ Label: First Input Folder │
│ Type: Managed Folder │
│ Maps to: input1_folder │
│ │
│ [+ Add Input] │
│ Name: input2 │
│ Label: Second Input Folder │
│ Type: Managed Folder │
│ Maps to: input2_folder │
│ │
│ Outputs: │
│ [+ Add Output] │
│ Name: output │
│ Label: Matched Results │
│ Type: Managed Folder │
│ Maps to: output_folder │
└─────────────────────────────────────┘

```

---

### 6.4: Link Scenario

```

┌─────────────────────────────────────┐
│ Scenario │
├─────────────────────────────────────┤
│ Select scenario: │
│ [build_fuzzy_results ▼] │
└─────────────────────────────────────┘

```

**This scenario runs when users execute the recipe.**

---

### 6.5: Configure Settings (User Parameters)

Map your project variables to user-facing form:

```

┌─────────────────────────────────────────────────────┐
│ Settings │
├─────────────────────────────────────────────────────┤
│ [+ Add Setting] │
│ │
│ Setting 1: │
│ Variable: logic_type │
│ Label: Matching Logic │
│ Type: Select │
│ Options: [Fuzzy Matching, De-duplication] │
│ Default: Fuzzy Matching │
│ │
│ Setting 2: │
│ Variable: id_column_1 │
│ Label: File 1 - ID Column │
│ Type: Text │
│ Default: id │
│ │
│ Setting 3: │
│ Variable: comparison_column_1 │
│ Label: File 1 - Comparison Column │
│ Type: Text │
│ Default: company_name │
│ │
│ Setting 4: │
│ Variable: id_column_2 │
│ Label: File 2 - ID Column │
│ Type: Text │
│ Default: id │
│ │
│ Setting 5: │
│ Variable: comparison_column_2 │
│ Label: File 2 - Comparison Column │
│ Type: Text │
│ Default: company_name │
│ │
│ Setting 6: │
│ Variable: threshold │
│ Label: Similarity Threshold (%) │
│ Type: Integer │
│ Default: 80 │
│ Min: 0 │
│ Max: 100 │
└─────────────────────────────────────────────────────┘

```

**Why**: These become form fields users fill when instantiating the recipe.

---

### 6.6: Publish

1. **Click "PUBLISH"** (top-right)

2. **Success message appears**

3. **Your recipe is now available system-wide!**

---

## 🎮 STEP 7: How Users Will Use Your Recipe

### User Workflow:

1. **Go to any project** → **Flow**

2. **Click "+ NEW RECIPE"**

3. **Find**: "Data Quality" → "Fuzzy Matching Tool"

4. **Configuration form appears**:

```

┌─────────────────────────────────────────────────────┐
│ Fuzzy Matching Tool │
├─────────────────────────────────────────────────────┤
│ Inputs: │
│ First Input Folder: [Select Folder ▼] │
│ Second Input Folder: [Select Folder ▼] │
│ │
│ Output: │
│ Matched Results: [new_folder_name] │
│ │
│ Settings: │
│ Matching Logic: [Fuzzy Matching ▼] │
│ │
│ File 1 - ID Column: [customer_id] │
│ File 1 - Comparison Column: [name] │
│ │
│ File 2 - ID Column: [vendor_id] │
│ File 2 - Comparison Column: [company] │
│ │
│ Similarity Threshold: [85] │
│ │
│ [CREATE RECIPE] │
└─────────────────────────────────────────────────────┘

```

5. **User fills form** → **Clicks "CREATE RECIPE"**

6. **Recipe appears in Flow** with their selected inputs/outputs

7. **User clicks "RUN"**

8. **Your scenario executes** → **Output folder contains results Excel file**

---

## 📊 Visual Flow Diagram

```

USER'S PROJECT:

[Their Folder 1] ──┐
├──→ [Fuzzy Matching Tool] ──→ [Results Folder]
[Their Folder 2] ──┘ (Your Recipe) (Excel file)
↓
[Build Scenario Runs]
↓
[Python Code Executes]

````

---

## ✅ Testing Checklist

### Before Publishing:

- [ ] Created all 3 folders (input1, input2, output)
- [ ] Created Python recipe with all code
- [ ] Set up 6 project variables with defaults
- [ ] Created scenario with build step
- [ ] Activated scenario
- [ ] Tested recipe manually (upload sample files)

### In Application Designer:

- [ ] Set recipe name and description
- [ ] Configured icon and category
- [ ] Mapped inputs (2 folders)
- [ ] Mapped output (1 folder)
- [ ] Linked scenario
- [ ] Configured all 6 settings
- [ ] Published application

### After Publishing:

- [ ] Recipe appears in recipe menu
- [ ] Can create instance in test project
- [ ] Settings form shows all parameters
- [ ] Can enter custom column names
- [ ] Recipe runs with custom values
- [ ] Recipe runs with default values (blank form)
- [ ] Output file generated correctly

---

## 🐛 Troubleshooting

### Issue 1: "Variable not found"

**Cause**: Variable name mismatch
**Fix**: Ensure variable names in Python code match exactly:

```python
variables.get('standard', {}).get('id_column_1', 'id')
                                  ↑ Must match variable name exactly
````

---

### Issue 2: "Folder ID not found" or "Cannot find folder"

**Cause**: Typo in folder ID or using ID from different Dataiku instance

**Fix Option 1** - Check folder ID in Flow:

1. Go to FlowRecipe inputs not found"

**Cause**: Recipe not properly connected to folders

**Fix**:

1. In Flow, click on your recipe
2. Check "Inputs" and "Outputs" tabs
3. Ensure 2 input folders and 1 output folder are connected
4. Recipe API automatically uses whatever is connected - no code changes needed!

**Cause**: User entered wrong column name or file doesn't have that column
**Fix**: Add better error message in Python code with available columns list

```python
if ID_COLUMN_1 not in df1.columns:
    available_cols = ', '.join(df1.columns.tolist())
    raise ValueError(f"Column '{ID_COLUMN_1}' not found in file 1.\n"
                     f"Available columns: {available_cols}")
```

---

### Issue 4: "No files found in folder"

### Issue 4: "No files found in folder"

**Cause**: User didn't upload files to input folders
**Fix**: Add validation check in Python code:

```python
if not input1_files:
    raise ValueError("No Excel/CSV files found in input1_folder. "
                     "Please upload at least one file.")
```

---

### Issue 5: "Scenario not executing"

**Fix**: The code already includes helpful error messages:

```python
if ID_COLUMN_1 not in df1.columns:
    raise ValueError(f"Column '{ID_COLUMN_1}' not found in file 1. "
                     f"Available: {list(df1.columns)}")
```

Tell user to check their custom variable values match actual column names.

### Tip 1: Support Multiple File Formats

Current code supports `.xlsx`, `.xls`, `.csv` - you can add more:

```python
if file_path.endswith(('.xlsx', '.xls', '.csv', '.parquet')):
```

**Fix**: The code already handles this with `read_file_from_folder()`:

```python
if not paths:
    raise ValueError("Input folder is empty. Please upload a file.")
```

User needs to upload files to connected folders before running recipe. = []
for file_path in input1_files:
if file_path.endswith('.xlsx'):
df = pd.read_excel(...)
dfs.append(df)
df1 = pd.concat(dfs, ignore_index=True)

````

### Tip 3: Add Validation

Add pre-checks before matching:

```python
if df1.empty:
    raise ValueError("Input file 1 is empty")
if len(df1) > 100000:
    print("Warning: Large dataset, may take time")
````

### Tip 4: Export Multiple Formats

Write both Excel and CSV:

```python
result_df.to_excel(...)
result_df.to_csv(...)
```

---

## 🎊 Success!

You've now created a fully functional Application-as-Recipe that:

- ✅ Accepts user input for column names
- ✅ Has sensible defaults (won't fail if empty)
- ✅ Processes files from folders
- ✅ Runs scenario automatically
- ✅ Can be used across all projects
- ✅ Provides clear user interface

**Show this to your senior** - they'll be impressed! 🚀
