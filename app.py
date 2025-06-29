import streamlit as st
import pandas as pd
import ahocorasick
import os
from collections import defaultdict
import requests
import io

# --- The Core Matching Logic (BiomarkerFinder Class) ---
# This class contains the updated, more robust logic from your Colab script.
class BiomarkerFinder:
    """
    A robust class that finds biomarkers using a case-insensitive, high-performance
    Aho-Corasick algorithm. It includes data validation, stop-word filtering,
    and whole-word boundary detection to improve accuracy.
    """
    def __init__(self, biomarker_dataframe, min_len=2):
        # Using st.write for logging in the Streamlit interface
        st.info("Initializing BiomarkerFinder...")
        
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 
            'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 
            'were', 'will', 'with', 'his'
        }
        
        self.biomarker_df = biomarker_dataframe
        
        self._validate_data()

        self.biomarker_df = self.biomarker_df.dropna(subset=['ID', 'Biomarker Name'])
        self.biomarker_db = {row['ID']: row.to_dict() for _, row in self.biomarker_df.drop_duplicates(subset=['ID']).iterrows()}

        st.success(f"Successfully loaded and validated biomarker data. Using {len(self.biomarker_db)} unique biomarkers.")
            
        self.automaton = self._build_automaton(min_len)
        st.info("Initialization complete. Matcher is ready.")

    def _validate_data(self):
        """Checks for duplicate IDs and warns the user in the app."""
        duplicates = self.biomarker_df[self.biomarker_df['ID'].duplicated(keep=False)]
        if not duplicates.empty:
            st.warning("⚠️ DATA QUALITY WARNING: Duplicate IDs found in biomarker.csv!")
            st.write("This can lead to incorrect or inconsistent matching. The script will use the FIRST entry found for each duplicate ID.")
            st.write("Affected IDs and their assigned names:")
            for_display = duplicates.groupby('ID')['Biomarker Name'].apply(list).reset_index()
            st.dataframe(for_display)

    def _build_automaton(self, min_len):
        """
        Builds a case-insensitive Aho-Corasick automaton, ignoring stop words.
        This version uses a stricter matching logic based on full terms only.
        """
        st.write("Building case-insensitive Aho-Corasick automaton for matching...")
        A = ahocorasick.Automaton()
        
        # Keep track of terms added to the automaton to resolve ambiguity (first seen wins)
        final_terms_map = {}

        # CORRECTED: Iterate over the entire dataframe, not just unique IDs, to capture all terms.
        # The final_terms_map will handle de-duplication of search terms.
        for _, row in self.biomarker_df.iterrows():
            concept_id = row['ID']
            
            # Combine biomarker name and synonyms into one list for processing
            terms_to_process = [str(row['Biomarker Name'])]
            synonyms = row.get('Exhaustive Synonyms')
            if pd.notna(synonyms):
                terms_to_process.extend(str(synonyms).split(','))
            
            for term in terms_to_process:
                term_stripped = term.strip()
                if self._is_valid_term(term_stripped):
                    term_lower = term_stripped.lower()
                    # Only add the term if it hasn't been mapped to an ID yet.
                    # This prevents overwrites and ensures the first seen ID for a term is kept.
                    if term_lower not in final_terms_map:
                        final_terms_map[term_lower] = (concept_id, term_stripped)
        
        # Now, add all the finalized terms to the automaton
        for term_lower, (concept_id, original_cased_term) in final_terms_map.items():
            A.add_word(term_lower, (concept_id, original_cased_term))

        A.make_automaton()
        st.write("Automaton build complete.")
        return A

    def _is_valid_term(self, term):
        """Helper function to check if a term is valid for matching."""
        return len(term) >= 2 and term.lower() not in self.stop_words

    def find_matches(self, text):
        """Processes text case-insensitively and returns a list of matched biomarkers."""
        if not self.automaton or self.biomarker_df.empty:
            return []
            
        all_matches = []
        for end_index, (concept_id, original_cased_term) in self.automaton.iter(text.lower()):
            start_index = end_index - len(original_cased_term) + 1
            actual_mention = text[start_index : end_index + 1]
            all_matches.append((start_index, end_index, concept_id, actual_mention))

        whole_word_matches = []
        for start, end, concept_id, term in all_matches:
            is_start_bound = (start == 0) or (not text[start - 1].isalnum())
            is_end_bound = (end + 1 == len(text)) or (not text[end + 1].isalnum())
            if is_start_bound and is_end_bound:
                whole_word_matches.append((start, end, concept_id, term))

        if not whole_word_matches:
            return []

        whole_word_matches.sort(key=lambda x: (x[0], - (x[1] - x[0]))) 
        
        longest_matches = []
        last_end = -1
        for match in whole_word_matches:
            start, end, _, _ = match
            if start > last_end:
                longest_matches.append(match)
                last_end = end

        results = []
        for start, end, concept_id, term in longest_matches:
            biomarker_info = self.biomarker_db.get(concept_id, {})
            results.append({
                "Mentioned Term": term,
                "Linked Biomarker": biomarker_info.get("Biomarker Name"),
                "ID": concept_id,
                "Type": biomarker_info.get("Type")
            })
        return results

# --- Streamlit App UI and Logic ---

st.set_page_config(page_title="Biomarker Extractor", layout="wide")
st.title("🔬 Biomarker Extractor from Clinical Text")

# Function to load data from GitHub, cached for performance
@st.cache_data
def load_data_from_github(url):
    """Fetches a CSV file from a public GitHub URL."""
    # Convert standard GitHub URL to raw URL
    if "github.com" in url and "raw.githubusercontent.com" not in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from GitHub: {e}")
        st.error("Please ensure the URL is correct and the repository is public.")
        return None

# --- Main App Logic ---

# Hardcoded URL to the biomarker CSV file
repo_url = "https://github.com/rajanpreets/mark_1/blob/6fd87d983a577ed9e3cbfbba07ba3c5a314d041b/biomarkers.csv"

# Sidebar for feedback link
st.sidebar.header("Feedback")
st.sidebar.info(
    "Have suggestions or found an issue? \n\n"
    "Please provide your feedback in our sheet: \n\n"
    "[Feedback Sheet](https://docs.google.com/spreadsheets/d/1q2MHXSZZraGUXd4fvyJAIn_QdVAYLYTEX7Qn_4j38-I/edit?usp=sharing)"
)

# Load data automatically from the hardcoded URL
biomarker_df = load_data_from_github(repo_url)

if biomarker_df is not None:
    # Initialize the finder once the data is loaded
    # The @st.cache_resource decorator ensures this only runs once
    @st.cache_resource
    def get_finder(df):
        return BiomarkerFinder(biomarker_dataframe=df)

    finder = get_finder(biomarker_df)
    
    st.markdown("---")
    
    st.header("Enter Clinical Text")
    
    default_text = (
        "Patient ID 78-B2 presented with elevated levels of Glycated hemoglobin.\n"
        "Lab results confirm high A1C and also show increased C-reactive protein.\n"
        "The presence of the enzyme ACE was also noted. We will monitor for changes in \n"
        "Alpha-fetoprotein (AFP) and Troponin I. Levels of Amyloid Beta 42 and amyloid-beta were also checked.\n"
        "The technician double-checked the results."
    )
    input_text = st.text_area("Paste your text here:", value=default_text, height=250)
    
    if st.button("Extract Biomarkers", type="primary"):
        if input_text:
            with st.spinner("Analyzing text..."):
                linked_biomarkers = finder.find_matches(input_text)
                
                unique_biomarkers = {}
                for biomarker in linked_biomarkers:
                    biomarker_id = biomarker.get('ID', 'N/A')
                    if biomarker_id not in unique_biomarkers:
                        unique_biomarkers[biomarker_id] = {
                            "Linked Biomarker": biomarker.get('Linked Biomarker'),
                            "ID": biomarker_id,
                            "Type": biomarker.get('Type'),
                            "Mentioned As": set()
                        }
                    unique_biomarkers[biomarker_id]["Mentioned As"].add(biomarker['Mentioned Term'])
                
                st.markdown("---")
                st.header("Results")
                
                if unique_biomarkers:
                    st.subheader("Summarized Biomarker Insights")
                    for biomarker_id, info in unique_biomarkers.items():
                        st.markdown(f"**- Biomarker:** {info['Linked Biomarker']} (ID: {info['ID']})")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Mentioned As:** `{'`, `'.join(info['Mentioned As'])}`")
                else:
                    st.info("No biomarkers from your list were found in the text.")
        else:
            st.warning("Please enter some text to analyze.")
else:
    # This message shows if the initial data load from GitHub fails
    st.error("Could not load the biomarker data. The application cannot proceed.")
