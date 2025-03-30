# Congressional Record and Bill Data Collection Pipeline

This directory contains the code for collecting, preprocessing, and labeling congressional speeches and bill data used in the research project. The pipeline consists of three main components:

1. **Data Collection**: Fetch speeches from the Congressional Record and bill information from the Congress.gov API
2. **Data Preprocessing**: Clean and align speech data with corresponding bill information
3. **Topic Labeling**: Use GPT to automatically identify the main political topic of each speech

## Setup Requirements

### API Keys
You'll need to create an `api_keys.py` file with the following keys:
```python
# For Congress.gov API
CONGRESS_API_KEY = "your_congress_api_key"
ALT_CONGRESS_API_KEY = "your_alternate_congress_api_key"  # Optional backup

# For OpenAI API (used in llm_label.py)
OPENAI_API_KEY = "your_openai_api_key"
```

### Python Dependencies
Required packages:
- pandas
- requests
- tqdm
- python-dateutil
- openai
- beautifulsoup4

## Pipeline Components

### 1. Data Preprocessing (`preprocess.py`)

This script fetches Congressional Record speech data from local files and retrieves detailed bill information from the Congress.gov API.

**Main Functions:**
- Loads Congressional Record speech data from local CSV files
- Preprocesses speech text to normalize formatting and remove procedural language
- Identifies speeches related to specific bills based on title matching
- Fetches detailed bill information from the Congress.gov API, including:
  - Policy areas and legislative subjects
  - Bill summaries
  - Committee information
- Matches speeches to their corresponding bills
- Outputs an enriched dataset for further analysis

**Usage:**
```bash
python preprocess.py [--data-dir DATA_DIR] [--output-file OUTPUT_FILE]
```

**Output:**
A CSV file containing speeches enriched with bill metadata.

### 2. Congress Member Data Collection (`get_congress_members.py`)

This script fetches information about members of Congress using the Congress.gov API, including their party affiliations and chamber.

**Main Functions:**
- Retrieves information for all members of a specific Congress (default: 117th)
- Extracts relevant information such as party affiliation, chamber, and state
- Formats member names to match the format used in the Congressional Record
- Outputs a CSV file with member information

**Usage:**
```bash
python get_congress_members.py
```

**Output:**
A CSV file with columns: 
- `speaker`: Member name in uppercase (LASTNAME)
- `chamber`: Chamber code (S for Senate, H for House)
- `party`: Party code (D for Democratic, R for Republican, I for Independent)
- `state`: Member's state

### 3. Topic Labeling with GPT (`llm_label.py`)

This script uses the OpenAI GPT API to automatically extract and label the main political topic from congressional speeches.

**Main Functions:**
- Takes the preprocessed data from `preprocess.py` as input
- Uses GPT to extract a short (2-5 word) topic for each speech
- The topic is designed to describe the main political issue in a way that can have a position taken on it
- Caches results to avoid unnecessary API calls
- Outputs a CSV file with the original data plus topic labels

**Usage:**
```bash
python llm_label.py [--input-file INPUT_FILE] [--output-file OUTPUT_FILE] [--data-dir DATA_DIR]
```

**Output:**
A CSV file with the following columns:
- Original speech data (document, speaker, chamber, etc.)
- Policy area from Congress.gov
- GPT-generated topic label
- Related metadata (legislative subjects, committees, summary)

## Full Data Collection Pipeline

The complete data collection process follows these steps:

1. Use `preprocess.py` to fetch speeches and bill data
2. Use `get_congress_members.py` to collect party affiliation information
3. Use `llm_label.py` to label each speech with its political topic

This creates a comprehensive dataset that can be used for various analyses, such as tracking political positions, identifying partisan differences, and examining issue polarization.

## Notes on Batch Processing

The `.slurm` files in this directory are batch job scripts for running the data collection pipeline on a compute cluster. These can be submitted using the `sbatch` command:

```bash
sbatch llm_label.slurm
```

These scripts are configured for systems that use Slurm as the job scheduler.
