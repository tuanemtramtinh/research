# Use Case Description Report: Renew Prescription

**Use Case Name**  
Renew Prescription

**Description**  
Allows a patient to request and renew prescriptions using the mobile platform.

**Primary Actor**  
Patient

**Problem Domain Context**  
Develop an integrated patient‑care platform that:

1. Provides patients with secure mobile access to their full medical record, including measurements, test results, and prescription renewal.  
2. Enables patients to share wearable‑device data and selectively grant or revoke access to healthcare providers.  
3. Allows doctors to view, edit, and annotate patient records, schedule appointments, request and track laboratory tests, and issue prescriptions while managing pharmacy inventory visibility.  
4. Permits nurses to edit and review records for clinical preparation.  
5. Supports receptionists with patient identification, appointment scheduling, and room assignment using doctor and lab availability.  
6. Equips laboratory technicians with a dashboard of pending tests, urgency, and requester details, along with the ability to input results and attach notes.  
7. Offers pharmacists the ability to verify patient ID, dosage, and prescription details.  
8. Gives system administrators the capability to configure user roles, enforce two‑step authentication, and adjust access rights across the system.

**Preconditions**  
- The patient is authenticated and logged into the mobile application.  
- The patient has an active prescription that is eligible for renewal.  
- The prescriber’s electronic signature and prescribing authority are enabled in the system.  
- Pharmacy inventory is up‑to‑date and the medication is in stock.

**Postconditions**  
- A renewal request is recorded in the patient’s electronic health record (EHR).  
- The prescriber receives a notification to review and sign the renewal.  
- Upon approval, the prescription is updated and made available for pharmacy dispensing.  
- The patient receives a confirmation notification.  

**Main Flow**  
1. **Patient** opens the mobile app and navigates to the “Prescriptions” section.  
2. **Patient** selects the prescription to renew and taps “Renew”.  
3. The app displays the prescription details (medication, dosage, refill history) for review.  
4. **Patient** confirms renewal request and submits.  
5. The system logs the renewal request and sends a notification to the prescribing **Doctor**.  
6. **Doctor** receives the notification in the clinician portal.  
7. **Doctor** reviews the patient’s current health status, recent labs, and medication adherence.  
8. **Doctor** approves or denies the renewal.  
9. If approved, the system updates the prescription record, decrements the remaining refills, and generates a new prescription ID.  
10. The system sends a confirmation to the **Pharmacist** with updated medication details.  
11. **Pharmacist** verifies patient identity, checks inventory, and processes the dispense.  
12. The patient receives an in‑app and email confirmation that the prescription is ready for pickup.  

**Alternative Flows**  
- **3a.** The patient chooses to request a different dosage or frequency.  
  1. The app captures the new dosage details.  
  2. The renewal request includes the dosage change and proceeds to step 5.  

- **7a.** The doctor identifies a contraindication or adverse reaction.  
  1. The doctor selects “Reject” and provides a reason in the comment field.  
  2. The system logs the rejection and sends a notification to the patient explaining the decision.  

- **9a.** The prescription renewal is pending until the next scheduled appointment.  
  1. The system delays the update until the appointment date, then repeats step 9.  

**Exceptions**  
- **5.** *Prescription not eligible for renewal*  
  1. The system displays an error message: “This prescription cannot be renewed due to policy restrictions.”  
  2. The flow ends; the patient is advised to contact their provider.  

- **6.** *Doctor not reachable*  
  1. The system sends a retry notification after 24 hours.  
  2. If still unresponsive, a support ticket is created for escalation.  

- **9.** *Medication out of stock*  
  1. The system notifies the pharmacist and the patient that the medication is unavailable.  
  2. The flow offers an alternative medication option if available; otherwise, the renewal is postponed.  

- **10.** *Pharmacist fails to verify patient ID*  
  1. The system logs the failure and halts dispensing.  
  2. The pharmacist must re‑verify identity or involve a supervisor before proceeding.  

---