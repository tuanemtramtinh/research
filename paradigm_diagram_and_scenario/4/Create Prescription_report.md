## Use Case Name  
**Create prescription**

## Description  
Doctors create electronic prescriptions for patients and transfer them to pharmacies.

## Primary Actor  
- Doctor

## Problem Domain Context  
The system must:  
1. Enable doctors to create, view, and electronically transfer prescriptions to pharmacies.  
2. Allow patients to view, edit personal information, and request new prescriptions through online or mobile appointment scheduling.  
3. Provide nurses, receptionists, and doctors with secure access to patients’ electronic medical records, including medical history, personal details, and notes.  
4. Support mobile device synchronization for doctors and patients to maintain consistent workflow across platforms.  
5. Allow pharmacists to view transferred prescriptions and manage medication dispensing.  
6. Grant system administrators the ability to centrally store all data, integrate electronic medical records, enforce GDPR and privacy controls, and manage user access levels.  
7. Supply systems analysts with anonymized clinical data, bed occupancy, room scheduling, and employee schedules for analytics.  
8. Record appointment start and end times for duration calculations.  
9. Support migration of data from legacy systems to the new platform.  
10. Implement secure authentication and authorization mechanisms to restrict system component access to authorized users.

## Preconditions  
- Doctor is authenticated and authorized.  
- Patient record exists in the system.  
- Patient has an active appointment or prescription request.  
- System has access to the latest drug database.

## Postconditions  
- Prescription is stored in the system.  
- Prescription is electronically transmitted to the selected pharmacy.  
- Confirmation is logged and displayed to the doctor.  
- Patient receives notification of the new prescription.

## Main Flow  
1. **Doctor** logs into the electronic health record (EHR) system.  
2. **Doctor** selects the patient’s record from the patient list.  
3. **Doctor** chooses the “Create Prescription” action.  
4. **System** displays the prescription form populated with the patient’s demographics and medical history.  
5. **Doctor** enters medication details (drug name, dosage, frequency, duration).  
6. **Doctor** selects the pharmacy to which the prescription will be sent.  
7. **Doctor** reviews the prescription summary for accuracy.  
8. **Doctor** confirms and submits the prescription.  
9. **System** validates all required fields and checks drug‑interaction and dosage constraints.  
10. **System** stores the prescription record in the database.  
11. **System** encrypts and transmits the prescription to the chosen pharmacy via HL7/FHIR API.  
12. **System** logs the transmission event and updates the patient’s prescription history.  
13. **System** displays a success message to the **Doctor** and sends an email/SMS notification to the patient.  

## Alternative Flows  
- **AF1 (Step 5 – Drug Selection)**  
  1. **Doctor** selects a drug not listed in the drug database.  
  2. **System** prompts the doctor to enter drug details manually or to add the drug to the database.  
  3. **Doctor** confirms the manual entry and proceeds.  

- **AF2 (Step 9 – Validation Failure)**  
  1. **System** detects a drug interaction or dosage error.  
  2. **System** displays a warning message and highlights the problematic field.  
  3. **Doctor** chooses to adjust the prescription or override the warning with justification.  

- **AF3 (Step 11 – Transmission Failure)**  
  1. **System** fails to transmit the prescription to the pharmacy.  
  2. **System** queues the prescription for retry and displays an error notification to the **Doctor**.  
  3. **Doctor** may attempt retransmission after a brief delay.

## Exceptions  
- **E1 (Step 1 – Authentication Failure)**  
  1. **System** denies access and displays an authentication error.  
  2. **Doctor** must re‑authenticate or contact support.  

- **E2 (Step 4 – Prescription Form Load Error)**  
  1. **System** cannot load patient data due to database connectivity issues.  
  2. **System** logs the error, displays a retry option, and informs the **Doctor** of the issue.  

- **E3 (Step 10 – Data Save Failure)**  
  1. **System** encounters a write error when storing the prescription.  
  2. **System** rolls back any partial changes, logs the failure, and prompts the **Doctor** to retry.  

- **E4 (Step 11 – Transmission Timeout)**  
  1. **System** does not receive acknowledgment from the pharmacy within the timeout period.  
  2. **System** queues the prescription, logs the timeout, and notifies the **Doctor** of the pending status.