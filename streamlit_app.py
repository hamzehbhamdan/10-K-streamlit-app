import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import psycopg2

sections = [
    "1 - Business",
    "1A - Risk Factors",
    "1B - Unresolved Staff Comments",
    "1C - Cybersecurity",
    "2 - Properties",
    "3 - Legal Proceedings",
    "4 - Mine Safety Disclosures",
    "5 - Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6 - Selected Financial Data (prior to February 2021)",
    "7 - Management’s Discussion and Analysis of Financial Condition and Results of Operations",
    "7A - Quantitative and Qualitative Disclosures about Market Risk",
    "9 - Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A - Controls and Procedures",
    "9B - Other Information",
    "10 - Directors, Executive Officers and Corporate Governance",
    "11 - Executive Compensation",
    "12 - Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13 - Certain Relationships and Related Transactions, and Director Independence",
    "14 - Principal Accountant Fees and Services",
]

cols = ['Embeddings', '1_Embeddings', '1A_Embeddings', '1B_Embeddings', '2_Embeddings', '3_Embeddings', '4_Embeddings', '5_Embeddings', '6_Embeddings', '7_Embeddings', '7A_Embeddings', '9A_Embeddings', '9B_Embeddings', '10_Embeddings', '11_Embeddings', '12_Embeddings', '13_Embeddings', '14_Embeddings']
titles = ['Entire 10-K File', 'Section 1: Business', 'Section 1A: Risk Factors', 'Section 1B: Unresolved Staff Comments', 'Section 2: Properties', 'Section 3: Legal Proceedings', 'Section 4: Mine Safety Disclosures', 'Section 5: Market for Registrant’s Common Equity, Related Stockholder\n Matters and Issuer Purchases of Equity Securities', 'Section 6: Selected Financial Data', 'Section 7: Management’s Discussion and Analysis of\n Financial Condition and Results of Operations', 'Section 7A: Quantitative and Qualitative Disclosures about Market Risk', 'Section 9A: Controls and Procedures', 'Section 9B: Other Information', 'Section 10: Directors, Executive Officers and Corporate Governance', 'Section 11: Executive Compensation', 'Section 12: Security Ownership of Certain Beneficial Owners\n and Management and Related Stockholder Matters', 'Section 13: Certain Relationships and Related Transactions,\n and Director Independence', 'Section 14: Principal Accountant Fees and Services']

def extract_np_array(vector):
    vector = vector.replace('\n', '').replace('  ', ' ').replace('  ', ' ')
    vector = vector[2:-2]
    vector = vector.split(' ')
    vector = np.array(vector)
    vector = vector.flatten()
    vector = [float(num) for num in vector if num != '']
    return vector

def extract_np_arrays(columns_selected_idx, df, companies, idx1, idx2, type='Averaged'):
    if type == 'Concatenated':
        company1 = []
        company2 = []
        for idx in columns_selected_idx:
            company1.extend(extract_np_array(df.loc[df['Ticker'] == companies[idx1], cols[idx]].values[0]))
            company2.extend(extract_np_array(df.loc[df['Ticker'] == companies[idx2], cols[idx]].values[0]))
        return np.array(company1), np.array(company2)
    elif type == 'Averaged':
        company1 = []
        company2 = []
        for idx in columns_selected_idx:
            company1.append(extract_np_array(df.loc[df['Ticker'] == companies[idx1], cols[idx]].values[0]))
            company2.append(extract_np_array(df.loc[df['Ticker'] == companies[idx2], cols[idx]].values[0]))
        extended_embedding1 = np.mean(company1, axis=0)
        extended_embedding2 = np.mean(company2, axis=0)
        return extended_embedding1, extended_embedding2

@st.cache_data
def load_data(query, params=()):
    conn = psycopg2.connect(
        dbname='dbcje502ldqf5d', 
        user='ub8msocfao9jk5', 
        password='pe88f4d04d5995ee5b05b92782ab6428b7b7c739751166adc4aebc3ca69cd826c', 
        host='c11ai4tgvdcf54.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com', 
        port='5432'
    )

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def display_cosine_similarity_for_two(companies, df):
    st.write(f"Cosine Similarity for {companies[0]} and {companies[1]} by 10-K Filing section.")
    df = df[df['Ticker'].isin(companies)]
    similarities = {}
    for idx, col in enumerate(cols):
        similarity = 1 - cosine(extract_np_array(df.loc[df['Ticker'] == companies[0], col].values[0]), extract_np_array(df.loc[df['Ticker'] == companies[1], col].values[0]))
        similarities[titles[idx]] = similarity
        
    similarity_df = pd.DataFrame(list(similarities.items()), columns=["Section", "Cosine Similarity"])

    norm = plt.Normalize(similarity_df['Cosine Similarity'].min(), similarity_df['Cosine Similarity'].max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    similarity_df['Color'] = similarity_df['Cosine Similarity'].apply(lambda x: sm.to_rgba(x))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x="Cosine Similarity", y="Section", data=similarity_df, palette=similarity_df['Color'].tolist(), ax=ax)
    ax.set_title(f"Cosine Similarity for Sections of 10-K Filings between {companies[0]} and {companies[1]}")
    
    min_value = similarity_df['Cosine Similarity'].min()
    ax.set_xlim(min_value - 0.05, 1.0)

    ticks_values = np.arange(0, 1.05, 0.05)
    ticks = [tick for tick in ticks_values if min_value - 0.05 <= tick <= 1.0]
    ax.set_xticks(ticks)

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Cosine Similarity')
    
    st.pyplot(fig)

def display_cosine_similarity_matrix(companies, df, threshold=None, index_selected=[0]):
    if index_selected == []:
        index_selected = [0]
    companies_text = ', '.join(companies)
    st.write(f"Cosine Similarity Matrix for {companies_text}.")
    similarity_matrix = np.zeros((len(companies), len(companies)))
    for i in range(len(companies)):
        for j in range(len(companies)):
            if i != j:
                company1array, company2array = extract_np_arrays(columns_selected_idx=index_selected, df=df, companies=companies, idx1=i, idx2=j)
                similarity = 1 - cosine(company1array, company2array)
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
                if threshold is not None:
                    if similarity < threshold:
                        similarity_matrix[i, j] = similarity_matrix[j, i] = 0
            else:
                similarity_matrix[i, j] = np.nan
    
    similarity_df = pd.DataFrame(similarity_matrix, index=companies, columns=companies)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=False, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 10}, cbar_kws={"label": "Cosine Similarity"})
    ax.set_title("Cosine Similarity Matrix")
    st.pyplot(fig)

st.title("Similarity of Company's 10-K Filings")

st.write('This software shows similarity of companies based on their most recent 10-K filing (last updated May 2024). Using SEC BERT to tokenize and embed data, I measure similarity using cosine similarity. An analysis of accuracy with S&P 100 companies is shown in the paper below.')

with open('10-K Embeddings.pdf', 'rb') as file:
    file_data = file.read()

st.write('The code behind this project is available at https://github.com/hamzehbhamdan/10-K-Text-Clustering/.')

st.download_button(label='Paper PDF', data=file_data, file_name='10-K Embeddings.pdf', mime="application/pdf")

company_names = st.text_input("Enter company tickers separated by commas.", value='AAPL, META, AMZN, WMT').split(',')

if len(company_names) > 1:
    company_names = [name.strip().upper() for name in company_names]
    placeholders = ', '.join(['%s'] * len(company_names))
    query = f'SELECT * FROM filing10kdata WHERE filing10kdata."Ticker" IN ({placeholders})'
    filtered_df = load_data(query, params=company_names)
    if len(filtered_df) == len(company_names):
        if len(company_names) == 2:
            display_cosine_similarity_for_two(company_names, filtered_df)
        else:
            sections_selected = st.multiselect("Which section of the 10-K filing would you like to compare?", titles, default=['Entire 10-K File'])                
            idx_list = []
            for sec in sections_selected:
                idx_list.append(titles.index(sec))
            if 'Entire 10-K File' in sections_selected:
                idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            display_cosine_similarity_matrix(companies=company_names, df=filtered_df, index_selected=idx_list)
    else:
        st.write("Some company tickers are not found in the sample data. Please try again.")
else:
    st.write("Please enter at least two company tickers.")
