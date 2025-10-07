#!/usr/bin/env python3
"""Analyze synthetic healthcare dataset distribution"""

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('dbx/synthetic_data/output/synthetic_data_v1.csv')

print("="*70)
print("SYNTHETIC HEALTHCARE DATASET DISTRIBUTION ANALYSIS")
print("="*70)

print(f"\n1. DATASET SIZE:")
print(f"   Total records: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")

print(f"\n2. PROVIDER TYPE DISTRIBUTION:")
provider_dist = df['Provider_Type'].value_counts()
for ptype, count in provider_dist.items():
    print(f"   {ptype:20s}: {count:8,} ({count/len(df)*100:5.2f}%)")

print(f"\n3. RISK SCORE DISTRIBUTION:")
risk_dist = df['Risk_Score'].value_counts().sort_index()
for risk, count in risk_dist.items():
    print(f"   Risk Score {risk:2d}: {count:8,} ({count/len(df)*100:5.2f}%)")

print(f"\n4. CLAIM AMOUNT STATISTICS:")
print(f"   Mean:       ${df['Claim_Amount'].mean():,.2f}")
print(f"   Median:     ${df['Claim_Amount'].median():,.2f}")
print(f"   Std Dev:    ${df['Claim_Amount'].std():,.2f}")
print(f"   Min:        ${df['Claim_Amount'].min():,.2f}")
print(f"   Max:        ${df['Claim_Amount'].max():,.2f}")
print(f"   25th pct:   ${df['Claim_Amount'].quantile(0.25):,.2f}")
print(f"   75th pct:   ${df['Claim_Amount'].quantile(0.75):,.2f}")
print(f"   95th pct:   ${df['Claim_Amount'].quantile(0.95):,.2f}")

print(f"\n5. ADVERSE ACTIONS DISTRIBUTION:")
adverse_dist = df['Adverse_Actions'].value_counts()
for action, count in adverse_dist.items():
    print(f"   {action:20s}: {count:8,} ({count/len(df)*100:5.2f}%)")

print(f"\n6. BOARD CERTIFICATION:")
cert_dist = df['Board_Certification'].value_counts()
for cert, count in cert_dist.items():
    print(f"   {cert:20s}: {count:8,} ({count/len(df)*100:5.2f}%)")

print(f"\n7. REASSIGNMENT OF BENEFITS:")
reassign_dist = df['Reassignment_Of_Benefits'].value_counts()
for r, count in reassign_dist.items():
    print(f"   {r:20s}: {count:8,} ({count/len(df)*100:5.2f}%)")

print(f"\n8. TOP 10 LICENSE STATES:")
state_dist = df['License_State'].value_counts().head(10)
for state, count in state_dist.items():
    print(f"   {state:5s}: {count:8,} ({count/len(df)*100:5.2f}%)")

print(f"\n9. TOP 10 SPECIALTY CODES:")
specialty_dist = df['Specialty_Code'].value_counts().head(10)
for spec, count in specialty_dist.items():
    print(f"   {str(spec):5}: {count:8,} ({count/len(df)*100:5.2f}%)")

print(f"\n10. BILLING AGENCY:")
has_billing = df['Billing_Agency'].notna().sum()
no_billing = df['Billing_Agency'].isna().sum()
print(f"   Has Billing Agency:  {has_billing:8,} ({has_billing/len(df)*100:5.2f}%)")
print(f"   No Billing Agency:   {no_billing:8,} ({no_billing/len(df)*100:5.2f}%)")

print(f"\n11. DUPLICATE ANALYSIS:")
ssn_dups = df[df['SSN'].notna()].duplicated(subset=['SSN'], keep=False).sum()
ein_dups = df[df['ssn_ein'].notna()].duplicated(subset=['ssn_ein'], keep=False).sum()
bank_dups = df.duplicated(subset=['BANK'], keep=False).sum()
print(f"   Duplicate SSN:       {ssn_dups:8,} ({ssn_dups/len(df)*100:5.2f}%)")
print(f"   Duplicate EIN:       {ein_dups:8,} ({ein_dups/len(df)*100:5.2f}%)")
print(f"   Shared Bank Accts:   {bank_dups:8,} ({bank_dups/len(df)*100:5.2f}%)")

# Create fraud labels
fraud_indicators = []
fraud_indicators.append(df['Risk_Score'] > 10)
fraud_indicators.append(df['Adverse_Actions'].isin(['Malpractice', 'Suspension']))
ssn_dups_mask = df[df['SSN'].notna()].duplicated(subset=['SSN'], keep=False)
ein_dups_mask = df[df['ssn_ein'].notna()].duplicated(subset=['ssn_ein'], keep=False)
fraud_indicators.append(ssn_dups_mask | ein_dups_mask)
bank_dups_mask = df.duplicated(subset=['BANK'], keep=False)
fraud_indicators.append(bank_dups_mask)
high_claims = df['Claim_Amount'] > df['Claim_Amount'].quantile(0.95)
fraud_indicators.append(high_claims)
is_fraud = np.logical_or.reduce(fraud_indicators).astype(int)

print(f"\n12. FRAUD LABEL DISTRIBUTION (Based on indicators):")
fraud_count = is_fraud.sum()
legit_count = len(df) - fraud_count
print(f"   Fraudulent:          {fraud_count:8,} ({fraud_count/len(df)*100:5.2f}%)")
print(f"   Legitimate:          {legit_count:8,} ({legit_count/len(df)*100:5.2f}%)")

print(f"\n   Fraud Breakdown by Indicator:")
print(f"   - High Risk Score (>10):     {(df['Risk_Score'] > 10).sum():8,}")
print(f"   - Adverse Actions:           {df['Adverse_Actions'].isin(['Malpractice', 'Suspension']).sum():8,}")
print(f"   - Duplicate SSN/EIN:         {(ssn_dups_mask | ein_dups_mask).sum():8,}")
print(f"   - Shared Bank Accounts:      {bank_dups_mask.sum():8,}")
print(f"   - High Claims (top 5%):      {high_claims.sum():8,}")

print(f"\n13. ENROLLMENT DATE RANGE:")
enrollment_dates = pd.to_datetime(df['Enrollment_Date'])
print(f"   Earliest: {enrollment_dates.min()}")
print(f"   Latest:   {enrollment_dates.max()}")
print(f"   Span:     {(enrollment_dates.max() - enrollment_dates.min()).days} days ({(enrollment_dates.max() - enrollment_dates.min()).days/365:.1f} years)")

print(f"\n14. GENDER DISTRIBUTION (Individuals only):")
gender_dist = df[df['Provider_Type'] == 'Individual']['Gender'].value_counts()
ind_total = len(df[df['Provider_Type'] == 'Individual'])
for gender, count in gender_dist.items():
    print(f"   {gender:20s}: {count:8,} ({count/ind_total*100:5.2f}%)")

print(f"\n15. OWNERSHIP TYPE DISTRIBUTION (Organizations only):")
if 'Ownership_Type' in df.columns:
    owner_dist = df[df['Provider_Type'] == 'Organization']['Ownership_Type'].value_counts()
    org_total = len(df[df['Provider_Type'] == 'Organization'])
    for owner, count in owner_dist.items():
        print(f"   {owner:20s}: {count:8,} ({count/org_total*100:5.2f}%)")

print(f"\n16. UNIQUE VALUES:")
print(f"   Unique Banks:        {df['BANK'].nunique():8,}")
print(f"   Unique Billing Agencies: {df['Billing_Agency'].nunique():8,}")
print(f"   Unique States:       {df['License_State'].nunique():8,}")
print(f"   Unique Specialties:  {df['Specialty_Code'].nunique():8,}")

print("\n" + "="*70)
