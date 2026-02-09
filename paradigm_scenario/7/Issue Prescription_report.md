**Use Case Name:**  
Issue Prescription

**Description:**  
Permits a doctor to generate prescriptions and manage pharmacy inventory visibility.

**Primary Actor:**  
Doctor

**Problem Domain Context:**  
Develop an integrated patient‑care platform that  
1. provides patients with secure mobile access to their full medical record, including measurement, test results and prescription renewal;  
2. enables patients to share wearable‑device data and selectively grant or revoke access to healthcare providers;  
3. allows doctors to view, edit and annotate patient records, schedule appointments, request and track laboratory tests, and issue prescriptions while managing pharmacy inventory visibility;  
4. permits nurses to edit and review records for clinical preparation;  
5. supports receptionists with patient identification, appointment scheduling, and room assignment using doctor and lab availability;  
6. equips laboratory technicians with a dashboard of pending tests, urgency, and requester details, along with the ability to input results and attach notes;  
7. offers pharmacists the ability to verify patient ID, dosage, and prescription details; and  
8. gives system administrators the capability to configure user roles, enforce two‑step authentication, and adjust access rights across the system.

**Preconditions:**  
None.

**Postconditions:**  
The prescription is stored in the patient’s electronic record, a digital copy is sent to the selected pharmacy, and pharmacy inventory levels are updated accordingly.

**Main Flow:**  
1. **Doctor logs into the system** and selects *Issue Prescription* from the dashboard.  
2. **Doctor searches for the patient** by name, ID, or medical record number.  
3. **Doctor opens the patient’s record** and reviews relevant clinical data (history, lab results, allergies).  
4. **Doctor selects medication(s) to prescribe** from the integrated formulary, specifying dosage, frequency, duration, and any special instructions.  
5. **Doctor reviews pharmacy inventory visibility** to ensure selected medication is available at the chosen pharmacy.  
6. **Doctor confirms the prescription** and optionally adds a note or annotation.  
7. **System generates a digital prescription** (PDF/HL7 FHIR bundle) and records it in the patient’s medical record.  
8. **System sends the prescription electronically** to the selected pharmacy via the integrated pharmacy network.  
9. **Pharmacy receives the prescription**, verifies patient ID and dosage, and updates inventory levels.  
10. **System logs the transaction** and updates audit trails for compliance.

**Alternative Flows:**  
1.1 **Medication not in inventory** – The system flags the medication as unavailable; the doctor selects an alternate drug or schedules an inventory restock.  
2.1 **Patient data incomplete** – The doctor is prompted to obtain missing information (e.g., allergy history) before proceeding.  
5.1 **Pharmacy denies prescription** – The system notifies the doctor, logs the denial, and suggests alternative pharmacies or medications.

**Exceptions:**  
1.1 **Login failure** – System prompts for password reset or multi‑factor authentication.  
3.1 **Patient record not found** – System displays an error and offers a search retry or new patient creation.  
6.1 **Invalid dosage or frequency** – System validates against formulary rules and rejects the entry, prompting correction.  
8.1 **Electronic transmission error** – System retries transmission up to three times, then flags the prescription for manual follow‑up.  
9.1 **Pharmacy inventory mismatch** – System alerts pharmacy staff to reconcile stock and notifies the prescribing doctor.