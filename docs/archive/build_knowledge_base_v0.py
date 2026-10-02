import pandas as pd
from pathlib import Path

services = [
    {
        "title": "Equivalency of Qualifications Issued Outside the Kingdom of Saudi Arabia",
        "agency": "Ministry of Education",
        "category": "Education and Training",
        "description": "Electronic service for requesting equivalency of university academic qualifications obtained outside Saudi Arabia. Applicants use the certificate equivalency portal, select the degree, enter certificate and previous academic degree information, and attach required documents.",
        "official_url": "https://my.gov.sa/en/services/18656"
    },
    {
        "title": "Instant Medical Consultation Service",
        "agency": "Ministry of Health",
        "category": "Health",
        "description": "Provides remote medical consultations with Ministry of Health approved doctors. Users access the service through Sehhaty, enter information about their medical condition, and request an instant consultation.",
        "official_url": "https://my.gov.sa/en/services/236898"
    },
    {
        "title": "Booking Medical Appointment",
        "agency": "Ministry of Health",
        "category": "Health",
        "description": "Allows users to book healthcare appointments, including healthcare center visits and remote consultations. Users can also manage, cancel, or reschedule appointments.",
        "official_url": "https://my.gov.sa/en/services/237229"
    },
    {
        "title": "View the Hospitals and Health Centers Directory",
        "agency": "Ministry of Health",
        "category": "Health",
        "description": "Allows users to search for hospitals, health centers and accredited pharmacies in Saudi Arabia using an interactive map and view facility locations and information.",
        "official_url": "https://my.gov.sa/en/services/237280"
    },
    {
        "title": "Suggestions and Complaints Services (937 Services)",
        "agency": "Ministry of Health",
        "category": "Health",
        "description": "Allows beneficiaries to electronically submit suggestions, complaints or reports to the Ministry of Health and track submitted tickets.",
        "official_url": "https://my.gov.sa/en/services/237232"
    },
    {
        "title": "Treatment Request Department System",
        "agency": "Ministry of Health",
        "category": "Health",
        "description": "Electronic system for managing treatment requests through the Unified Medical Referral Platform, including in-country treatment, overseas treatment and other treatment request pathways.",
        "official_url": "https://my.gov.sa/en/services/259198"
    },
    {
        "title": "Transaction Inquiry Service",
        "agency": "Ministry of Health",
        "category": "Health",
        "description": "Allows beneficiaries to electronically review the status of transactions previously submitted to Ministry of Health systems.",
        "official_url": "https://my.gov.sa/en/services/266208"
    },
    {
        "title": "Health Practitioner Training Program",
        "agency": "Ministry of Health",
        "category": "Employment and Training",
        "description": "Training program for eligible Saudi health practitioners. Applicants access the Ministry of Health portal and apply according to the program requirements.",
        "official_url": "https://my.gov.sa/en/services/272640"
    },
    {
        "title": "Book an Appointment Civil Affairs",
        "agency": "Ministry of Interior",
        "category": "Personal Documents",
        "description": "Allows beneficiaries to electronically book a new Civil Affairs appointment or modify an existing appointment and review service requirements before visiting the office.",
        "official_url": "https://my.gov.sa/en/services/2771538"
    },
    {
        "title": "Public Security Appointments",
        "agency": "Ministry of Interior",
        "category": "Safety and Interior Services",
        "description": "Allows citizens and residents to electronically book appointments with Public Security and Forensic Evidence departments through Absher.",
        "official_url": "https://my.gov.sa/en/services/538678"
    },
    {
        "title": "Inquire about existing commercial agencies",
        "agency": "Ministry of Commerce",
        "category": "Business and Entrepreneurship",
        "description": "Allows beneficiaries to electronically search and inquire about commercial agencies registered with the Ministry of Commerce.",
        "official_url": "https://my.gov.sa/en/services/736209"
    },
    {
        "title": "Digital Branch Services",
        "agency": "Ministry of Commerce",
        "category": "Business and Entrepreneurship",
        "description": "Allows businesses to electronically request Ministry of Commerce services requiring employee verification without visiting a physical service center.",
        "official_url": "https://my.gov.sa/en/services/240830"
    },
    {
        "title": "E-Authorization",
        "agency": "Ministry of Commerce",
        "category": "Business and Entrepreneurship",
        "description": "Allows establishment owners and company managers to electronically authorize individuals for selected services and manage their authorizations.",
        "official_url": "https://my.gov.sa/en/services/18866"
    },
    {
        "title": "Document Verification Service",
        "agency": "Ministry of Commerce",
        "category": "Business and Entrepreneurship",
        "description": "Allows beneficiaries to electronically verify documents associated with commercial registers, certificates of origin and discount licenses.",
        "official_url": "https://my.gov.sa/en/services/18821"
    },
    {
        "title": "Establish a Limited Liability Company",
        "agency": "Ministry of Commerce",
        "category": "Business and Entrepreneurship",
        "description": "Electronic Saudi Business Center service for establishing a limited liability company and completing related business registrations electronically.",
        "official_url": "https://my.gov.sa/en/services/18740"
    },
    {
        "title": "Establish a Simple Company",
        "agency": "Ministry of Commerce",
        "category": "Business and Entrepreneurship",
        "description": "Electronic service through the Saudi Business Center for establishing a simple limited partnership and completing associated business registrations.",
        "official_url": "https://my.gov.sa/en/services/18743"
    },
    {
        "title": "Renewal of Registration of a Commercial Agency",
        "agency": "Ministry of Commerce",
        "category": "Business and Entrepreneurship",
        "description": "Allows beneficiaries to electronically apply to renew the registration of a commercial agency and receive the renewed certificate.",
        "official_url": "https://my.gov.sa/en/services/18917"
    },
    {
        "title": "Search Licensed Consulting Professionals",
        "agency": "Ministry of Commerce",
        "category": "Business and Entrepreneurship",
        "description": "Allows users to search the database of people licensed to practice consulting professions and view license information.",
        "official_url": "https://my.gov.sa/en/services/18830"
    }
]

df = pd.DataFrame(services)

Path("data").mkdir(exist_ok=True)

df.insert(0, "service_id", range(1, len(df) + 1))
df["source"] = "GOV.SA National Platform"
df["language"] = "English"

df.to_csv(
    "data/services.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDALIL AI KNOWLEDGE BASE CREATED")
print("--------------------------------")
print("Verified seed records:", len(df))
print("Agencies:", df["agency"].nunique())
print("Categories:", df["category"].nunique())
print("Saved: data/services.csv")
print("--------------------------------")