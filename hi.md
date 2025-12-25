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

### 3.2: Write Recipe Code

The Python editor opens. **Replace all code** with this:

```python
"""
Fuzzy Matching Recipe
=====================
Reads Excel files from two input folders, performs fuzzy matching,
and writes results to output folder.
"""

import dataiku
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
# GET PROJECT VARIABLES (User inputs or defaults)
# ====================
client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())
variables = project.get_variables()

# Get user-defined columns or use defaults
ID_COLUMN_1 = variables.get('standard', {}).get('id_column_1', 'id')
COMPARISON_COLUMN_1 = variables.get('standard', {}).get('comparison_column_1', 'company_name')
ID_COLUMN_2 = variables.get('standard', {}).get('id_column_2', 'id')
COMPARISON_COLUMN_2 = variables.get('standard', {}).get('comparison_column_2', 'company_name')
THRESHOLD = int(variables.get('standard', {}).get('threshold', 80))
LOGIC_TYPE = variables.get('standard', {}).get('logic_type', 'Fuzzy Matching')

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


# ====================
# READ INPUT FOLDERS
# ====================
print("Reading input folders...")

input1_folder = dataiku.Folder("input1_folder")
input2_folder = dataiku.Folder("input2_folder")

# Get list of files in each folder
input1_files = input1_folder.list_paths_in_partition()
input2_files = input2_folder.list_paths_in_partition()

print(f"Input 1 files: {input1_files}")
print(f"Input 2 files: {input2_files}")

# Read first Excel file from each folder
df1 = None
df2 = None

for file_path in input1_files:
    if file_path.endswith(('.xlsx', '.xls', '.csv')):
        print(f"Reading file 1: {file_path}")
        with input1_folder.get_download_stream(file_path) as stream:
            if file_path.endswith('.csv'):
                df1 = pd.read_csv(stream)
            else:
                df1 = pd.read_excel(stream, engine='openpyxl')
        break

for file_path in input2_files:
    if file_path.endswith(('.xlsx', '.xls', '.csv')):
        print(f"Reading file 2: {file_path}")
        with input2_folder.get_download_stream(file_path) as stream:
            if file_path.endswith('.csv'):
                df2 = pd.read_csv(stream)
            else:
                df2 = pd.read_excel(stream, engine='openpyxl')
        break

if df1 is None:
    raise ValueError("No Excel/CSV file found in input1_folder")

# For deduplication, use same file for both
if LOGIC_TYPE == 'De-duplication':
    df2 = df1
    ID_COLUMN_2 = ID_COLUMN_1
    COMPARISON_COLUMN_2 = COMPARISON_COLUMN_1
elif df2 is None:
    raise ValueError("No Excel/CSV file found in input2_folder")

print(f"Dataset 1 shape: {df1.shape}")
print(f"Dataset 2 shape: {df2.shape}")

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
output_filename = f"fuzzy_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
with output_folder.get_writer(output_filename) as writer:
    result_df.to_excel(writer, index=False, engine='openpyxl')

print(f"Results written to: {output_filename}")
print("Recipe completed successfully!")
```

---

### 3.3: Save Recipe

1. **Click "SAVE"** (top-left)
2. **Click "RUN"** to test it (will fail initially - that's okay, we need to set variables)

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
│ Application Header                  │
├─────────────────────────────────────┤
│ Recipe Name:                        │
│ [Fuzzy Matching Tool]               │
│                                     │
│ Description:                        │
│ [Intelligent fuzzy matching and...] │
│                                     │
│ Permission:                         │
│ [✓] All users can instantiate       │
└─────────────────────────────────────┘
```

---

### 6.3: Configure Recipe Definition

```
┌─────────────────────────────────────┐
│ Recipe Definition                   │
├─────────────────────────────────────┤
│ Icon: icon-search                   │
│ Category: Data Quality              │
│                                     │
│ Inputs:                             │
│   [+ Add Input]                     │
│   Name: input1                      │
│   Label: First Input Folder         │
│   Type: Managed Folder              │
│   Maps to: input1_folder            │
│                                     │
│   [+ Add Input]                     │
│   Name: input2                      │
│   Label: Second Input Folder        │
│   Type: Managed Folder              │
│   Maps to: input2_folder            │
│                                     │
│ Outputs:                            │
│   [+ Add Output]                    │
│   Name: output                      │
│   Label: Matched Results            │
│   Type: Managed Folder              │
│   Maps to: output_folder            │
└─────────────────────────────────────┘
```

---

### 6.4: Link Scenario

```
┌─────────────────────────────────────┐
│ Scenario                            │
├─────────────────────────────────────┤
│ Select scenario:                    │
│ [build_fuzzy_results ▼]             │
└─────────────────────────────────────┘
```

**This scenario runs when users execute the recipe.**

---

### 6.5: Configure Settings (User Parameters)

Map your project variables to user-facing form:

```
┌─────────────────────────────────────────────────────┐
│ Settings                                            │
├─────────────────────────────────────────────────────┤
│ [+ Add Setting]                                     │
│                                                     │
│ Setting 1:                                          │
│   Variable: logic_type                              │
│   Label: Matching Logic                            │
│   Type: Select                                      │
│   Options: [Fuzzy Matching, De-duplication]        │
│   Default: Fuzzy Matching                          │
│                                                     │
│ Setting 2:                                          │
│   Variable: id_column_1                             │
│   Label: File 1 - ID Column                        │
│   Type: Text                                        │
│   Default: id                                       │
│                                                     │
│ Setting 3:                                          │
│   Variable: comparison_column_1                     │
│   Label: File 1 - Comparison Column                │
│   Type: Text                                        │
│   Default: company_name                            │
│                                                     │
│ Setting 4:                                          │
│   Variable: id_column_2                             │
│   Label: File 2 - ID Column                        │
│   Type: Text                                        │
│   Default: id                                       │
│                                                     │
│ Setting 5:                                          │
│   Variable: comparison_column_2                     │
│   Label: File 2 - Comparison Column                │
│   Type: Text                                        │
│   Default: company_name                            │
│                                                     │
│ Setting 6:                                          │
│   Variable: threshold                               │
│   Label: Similarity Threshold (%)                  │
│   Type: Integer                                     │
│   Default: 80                                       │
│   Min: 0                                            │
│   Max: 100                                          │
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
   │ Fuzzy Matching Tool                                 │
   ├─────────────────────────────────────────────────────┤
   │ Inputs:                                             │
   │   First Input Folder:  [Select Folder ▼]           │
   │   Second Input Folder: [Select Folder ▼]           │
   │                                                     │
   │ Output:                                             │
   │   Matched Results:     [new_folder_name]           │
   │                                                     │
   │ Settings:                                           │
   │   Matching Logic: [Fuzzy Matching ▼]               │
   │                                                     │
   │   File 1 - ID Column: [customer_id]                │
   │   File 1 - Comparison Column: [name]               │
   │                                                     │
   │   File 2 - ID Column: [vendor_id]                  │
   │   File 2 - Comparison Column: [company]            │
   │                                                     │
   │   Similarity Threshold: [85]                       │
   │                                                     │
   │              [CREATE RECIPE]                        │
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
[Their Folder 2] ──┘         (Your Recipe)           (Excel file)
                              ↓
                        [Build Scenario Runs]
                              ↓
                        [Python Code Executes]
```

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
```

### Issue 2: "Column not found in dataframe"

**Cause**: User entered wrong column name
**Fix**: Add better error message in Python code with available columns list

### Issue 3: "No files found in folder"

**Cause**: User didn't upload files to folder
**Fix**: Add check in Python code:

```python
if not input1_files:
    raise ValueError("Please upload at least one Excel/CSV file to input1_folder")
```

### Issue 4: "Scenario not executing"

**Cause**: Scenario not activated
**Fix**: Go to Scenarios → Toggle "Active" ON

---

## 💡 Pro Tips

### Tip 1: Support Multiple File Formats

Current code supports `.xlsx`, `.xls`, `.csv` - you can add more:

```python
if file_path.endswith(('.xlsx', '.xls', '.csv', '.parquet')):
```

### Tip 2: Process Multiple Files

Modify code to loop through all files and concatenate:

```python
dfs = []
for file_path in input1_files:
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(...)
        dfs.append(df)
df1 = pd.concat(dfs, ignore_index=True)
```

### Tip 3: Add Validation

Add pre-checks before matching:

```python
if df1.empty:
    raise ValueError("Input file 1 is empty")
if len(df1) > 100000:
    print("Warning: Large dataset, may take time")
```

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
