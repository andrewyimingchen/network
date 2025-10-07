#!/usr/bin/env python3
"""
Unified Healthcare Data Generator
Combines NPI generation and synthetic healthcare claims data generation
"""

import csv
import random
import string
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker
from fuzzywuzzy import fuzz

# ============================================================================ #
# ---------------------------- NPI FILLER ------------------------------------ #
# ============================================================================ #

class NPISequence:
    """Iterator for NPI numbers from a reference CSV file"""

    def __init__(self, csv_path, start=0, end=None):
        self.npi_list = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.npi_list.append(row['NPI'])

        if end is None or end > len(self.npi_list):
            end = len(self.npi_list)

        self.start = start
        self.end = end
        self.index = self.start
        self.total = self.end - self.start

    def next(self):
        if self.index >= self.end:
            raise IndexError(f"Ran out of NPIs in reference_npi_list.csv for the specified range!")
        npi = self.npi_list[self.index]
        self.index += 1
        return npi


# Create a global NPI sequence object with default full range
npi_seq = None


def fill_npi(start=0, end=None):
    """Get next NPI from the sequence"""
    global npi_seq
    if npi_seq is None or npi_seq.start != start or npi_seq.end != (end if end is not None else len(npi_seq.npi_list)):
        npi_seq = NPISequence('./output/reference_npi_list.csv', start, end)
    return npi_seq.next()


def luhn_checksum(num_str: str) -> str:
    """Compute the Luhn check digit (used for NPI)."""
    digits = [int(x) for x in num_str]
    parity = len(digits) % 2
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - (total % 10)) % 10)


def generate_unique_prefixed_numbers(num_to_generate, prefix, length=10):
    """
    Generate a set of unique numbers as strings, each with a specified prefix and total length.

    Args:
        num_hashes (int): Number of unique numbers to generate.
        prefix (int or str): The starting digits of each number.
        length (int): Total length of each number (default is 10).

    Returns:
        list: List of unique numbers as strings.
    """
    prefix_str = str(prefix)
    if len(prefix_str) >= length:
        raise ValueError("Prefix length must be less than total length.")

    num_digits = length - len(prefix_str)
    min_value = 10**(num_digits - 1)
    max_value = 10**num_digits - 1

    if num_to_generate > (max_value - min_value + 1):
        raise ValueError("Requested more numbers than possible with the given length and prefix.")

    hashes = set()

    while len(hashes) < num_to_generate:
        suffix = str(random.randint(min_value, max_value)).zfill(num_digits)
        hashes.add(prefix_str + suffix)

    return list(hashes)


def generate_npi():
    """Generate a valid NPI number (commented out for now)"""
    # base = ''.join(random.choices(string.digits, k=9))
    # return base + luhn_checksum(base)
    pass


# ============================================================================ #
# ----------------------- HELPER FUNCTIONS ----------------------------------- #
# ============================================================================ #

def generate_ssn():
    """Generate a random SSN"""
    return f"{random.randint(100, 899):03d}-{random.randint(10, 99):02d}-{random.randint(1000, 9999):04d}"


def generate_ein():
    """Generate a random EIN"""
    return f"{random.randint(10, 99):02d}-{random.randint(1000000, 9999999):07d}"


def random_date(start_year=1930, end_year=2000):
    """Generate a random date of birth"""
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def future_date(years_ahead=5):
    """Generate a future date"""
    today = date.today()
    return today + timedelta(days=random.randint(30, 365*years_ahead))


def generate_state_license(state_abbr):
    """Generate a state license number"""
    return f"{state_abbr}{random.randint(100000, 999999):06d}"


def generate_dea_number():
    """Generate DEA number"""
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    digits = ''.join(random.choices(string.digits, k=7))
    return letters + digits


def random_specialty_code():
    """Return a random CMS specialty code"""
    # Subset of actual CMS codes
    codes = ["01", "02", "03", "06", "08", "10", "11", "20", "30"]  # Internal Med, General Surg, ...
    return random.choice(codes)


def random_phone():
    """Generate a random phone number"""
    return f"+{random.randint(200, 999):03d}-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}"


def random_accreditation_org():
    """Return a random accreditation organization"""
    return random.choice(["JCAHO", "URAC", "NCQA", "DNV", None])


def random_ownership():
    """Return a random ownership type"""
    return np.random.choice(list(OWNERSHIP_TYPE_DIST.keys()),
                           p=list(OWNERSHIP_TYPE_DIST.values()))


def similar_name(name):
    """Return a slightly modified name with ≥ 90 % fuzzy similarity."""
    if "," in name:  # Handle org names with commas
        name = name.replace(",", "")

    tokens = name.split()
    if len(tokens) > 2 and random.random() < 0.5:
        tokens.pop(1)  # Drop middle name/initial
    else:
        tokens[0] = tokens[0][0] + tokens[0]  # Duplicate first letter

    return " ".join(tokens)


def similar_email(email):
    """Return a similar email with fuzzy similarity"""
    user, domain = email.split("@")
    if "." in user and random.random() < 0.5:
        user = user.replace(".", "")
    else:
        user = user.replace(".", "_")

    return f"{user}@{domain}"


def generate_risk_score():
    """Return 0 with 90% probability, else a random int between 1 and 20."""
    if random.random() < 0.1:
        return random.randint(1, 20)
    else:
        return 0


def generate_claim_amount():
    """Return a random claim amount between $1,000 and $10,000,000."""
    return random.randint(1000, 10000000)


# ============================================================================ #
# ------------------------ USER PARAMETERS ----------------------------------- #
# ============================================================================ #

TOTAL_ROWS = 1_000_000  # 1000 thousand rows
DUPLICATE_RATE = 0.02   # 2 % of TOTAL_ROWS will receive ID duplicates
FUZZY_DUPLICATE_RATE = 0.02  # 2 % of TOTAL_ROWS will receive fuzzy dupes

# Distribution knobs (must sum to 1.0)
PROVIDER_TYPE_DIST = {
    "Individual": 0.70,
    "Organization": 0.30
}

BOARD_CERT_DIST = {
    "Yes": 0.80,
    "No": 0.20
}

OWNERSHIP_TYPE_DIST = {
    "Sole Proprietor": 0.40,
    "LLC": 0.30,
    "Corp": 0.30
}


# ============================================================================ #
# -------------------------- MAIN GENERATOR ---------------------------------- #
# ============================================================================ #

def main():
    """Main data generation function"""

    # Initialize Faker
    fake = Faker()
    Faker.seed(42)
    random.seed(42)
    np.random.seed(42)

    # Generate reference NPI list first
    print("Generating reference NPI list...")
    reference_npi_list = generate_unique_prefixed_numbers(1000000, 2, length=10)
    print(f"Generated {len(reference_npi_list)} NPIs with prefix '2' and length 10.")

    # Write the reference_npi_list to a CSV file
    with open('./output/reference_npi_list.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['NPI'])
        for npi in reference_npi_list:
            writer.writerow([npi])

    print(f"Wrote {len(reference_npi_list)} NPIs to reference_npi_list.csv")

    # -------------------------------------------------------- #
    # ---------------- ID DUPLICATIONS ----------------------- #
    # -------------------------------------------------------- #

    # Precompute donor indices for efficiency
    fuzzy_dup_count = int(TOTAL_ROWS * FUZZY_DUPLICATE_RATE)
    fuzzy_indices = random.sample(list(range(TOTAL_ROWS)), fuzzy_dup_count)

    donor_indices = []
    for idx in fuzzy_indices:
        # Avoid picking the same index
        donor_idx = random.choice([x for x in range(TOTAL_ROWS) if x != idx])
        donor_indices.append(donor_idx)

    # Vectorized update using zip for efficiency
    for idx, donor_idx in zip(fuzzy_indices, donor_indices):
        base_name = fake.name()  # Will be replaced later
        base_email = fake.email()

        # Try a fixed number of times to avoid infinite loops
        for _ in range(10):
            new_name = similar_name(base_name)
            if fuzz.ratio(base_name, new_name) >= 90 and new_name != base_name:
                break
        else:
            new_name = base_name  # fallback

        for _ in range(10):
            new_email = similar_email(base_email)
            if fuzz.ratio(base_email, new_email) >= 90 and new_email != base_email:
                break
        else:
            new_email = base_email  # fallback

    # Now includes the combined BANK column
    dup_fields_choices = ["SSN", "ssn_ein", "State_License_Number", "BANK"]

    dup_count = int(TOTAL_ROWS * DUPLICATE_RATE)
    dup_indices = random.sample(list(range(TOTAL_ROWS)), dup_count)

    donor_indices = []
    for idx in dup_indices:
        # pick a random donor different from idx
        donor_idx = random.choice([x for x in range(TOTAL_ROWS) if x != idx])
        donor_indices.append(donor_idx)
        field = random.choice(dup_fields_choices)

        if field == "BANK":
            # copy the two underlying fields
            pass  # Will be handled in record creation
        else:
            # Only copy if donor has a non-null value (e.g., EIN for orgs)
            pass  # Will be handled in record creation

    # -------------------------------------------------------- #
    # --------------- GENERATE RECORDS ----------------------- #
    # -------------------------------------------------------- #

    address_pool = [fake.address().replace("\n", ", ") for _ in range(10000)]
    phone_pool = [f"+{random.randint(200, 999):03d}-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}" for _ in range(10000)]

    fake = Faker()

    records = []

    for i in range(TOTAL_ROWS):
        provider_type = np.random.choice(list(PROVIDER_TYPE_DIST.keys()),
                                        p=list(PROVIDER_TYPE_DIST.values()))
        is_individual = (provider_type == "Individual")

        if is_individual:
            name = fake.name()
            ssn = generate_ssn()
            ein = None
            dob = random_date()
            gender = random.choice(["Male", "Female"])
        else:
            name = fake.company()
            ssn = None
            ein = generate_ein()
            dob = None
            gender = None

        # contact & license
        contact_email = fake.email()
        license_state = fake.state_abbr()
        license_num = generate_state_license(license_state)

        # generate account + routing
        account = ''.join(random.choices(string.digits, k=random.randint(8,12)))
        routing = ''.join(random.choices(string.digits, k=9))

        record = {
            "NPI": fill_npi(),
            "Provider_Type": provider_type,
            "Provider_Name": name,
            "SSN": ssn,
            "ssn_ein": ein,
            "Contact_Email": contact_email,
            "DOB": dob.strftime("%m/%d/%Y") if dob else None,
            "Gender": gender,
            "Contact_Phone": random.choice(phone_pool),
            "Practice_Address": random.choice(address_pool),
            "Mailing_Address": random.choice(address_pool),
            "State_License_Number": license_num,
            "License_State": license_state,
            "License_Expiration": future_date(6).strftime("%m/%d/%Y"),
            "DEA_Number": generate_dea_number() if is_individual else None,
            "Specialty_Code": random_specialty_code(),
            "Board_Certification": np.random.choice(["Yes", "No"],
                                                    p=list(BOARD_CERT_DIST.values())),
            "Accreditation_Org": random_accreditation_org(),
            "Accreditation_Exp": future_date(6).strftime("%m/%d/%Y"),
            "Ownership_Type": random_ownership(),
            "Adverse_Actions": random.choice(["None"]*9 + ["Malpractice", "Suspension"]),
            "Bank_Account_Number": account,
            "Bank_Routing_Number": routing,
            "BANK": f"{account}-{routing}",
            "Billing_Agency": random.choice([fake.company(), None]),
            "Reassignment_Of_Benefits": random.choice(["Y","N"]),
            "Enrollment_Date": random_date(2005, 2023).strftime("%m/%d/%Y"),
            "Last_Updated": date.today().strftime("%m/%d/%Y"),
            "Risk_Score": generate_risk_score(),
            "Claim_Amount": generate_claim_amount()
        }
        records.append(record)

    df = pd.DataFrame(records)

    # COMMAND ----------
    print(f"\nDataframe info:")
    print(df.info())

    # COMMAND ----------
    print(f"\nDataframe head:")
    print(df.head())

    # -------------------------------------------------------- #
    # --------------- OUTPUT --------------------------------- #
    # -------------------------------------------------------- #

    output_file = "./output/synthetic_data_v1.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSynthetic data set generated: {output_file}")


if __name__ == "__main__":
    main()
