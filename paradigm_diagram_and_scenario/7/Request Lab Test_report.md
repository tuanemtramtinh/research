## Use Case Name: Request Lab Test

## Description
Allows a doctor to request laboratory tests and track their status.

## Primary Actor
Doctor

## Problem Domain Context
Develop an integrated patient‑care platform that:
1. Provides patients with secure mobile access to their full medical record, including measurement, test results and prescription renewal.  
2. Enables patients to share wearable‑device data and selectively grant or revoke access to healthcare providers.  
3. Allows doctors to view, edit and annotate patient records, schedule appointments, request and track laboratory tests, and issue prescriptions while managing pharmacy inventory visibility.  
4. Permits nurses to edit and review records for clinical preparation.  
5. Supports receptionists with patient identification, appointment scheduling, and room assignment using doctor and lab availability.  
6. Equips laboratory technicians with a dashboard of pending tests, urgency, and requester details, along with the ability to input results and attach notes.  
7. Offers pharmacists the ability to verify patient ID, dosage, and prescription details.  
8. Gives system administrators the capability to configure user roles, enforce two‑step authentication, and adjust access rights across the system.

## Preconditions
* The doctor is authenticated and authorized to request lab tests.  
* The patient’s record exists in the system.  
* The laboratory service is registered and its test catalog is available.  

## Postconditions
* A new lab test order is created in the system.  
* The order status is set to **Pending**.  
* The doctor receives a confirmation notification.  
* The laboratory technician receives a notification of the new order.

## Main Flow
1. **Doctor** logs into the patient‑care platform.  
2. **Doctor** selects a patient from the patient list.  
3. **Doctor** opens the patient’s record and navigates to the **Lab Tests** section.  
4. **Doctor** clicks **Request Lab Test**.  
5. The system displays the **Lab Test Request Form** pre‑filled with patient demographics.  
6. **Doctor** selects the required test(s) from the catalog, enters any additional notes, and specifies the priority (e.g., routine, urgent).  
7. **Doctor** reviews the request details and clicks **Submit**.  
8. The system validates the input, creates a new lab order, assigns a unique order ID, and sets status to **Pending**.  
9. The system logs the request event for audit purposes.  
10. The system notifies the laboratory technician(s) about the new order and updates the technician dashboard.  
11. **Doctor** receives a confirmation message and can view the order in the patient’s lab history.

## Alternative Flows
1. **(Step 6) Alternative Test Selection**  
   * The doctor selects **Allergy Panel** instead of the initially chosen test.  
2. **(Step 6) Additional Specimen Requirements**  
   * The doctor specifies a special specimen type (e.g., “Serum”) that triggers a note in the order.  
3. **(Step 7) Order Modification**  
   * Before submitting, the doctor cancels a test from the list and adds another.  

## Exceptions
1. **(Step 6) Catalog Unavailable**  
   * The lab test catalog is temporarily offline.  
   * *System Response:* Displays an error message, logs the incident, and suggests retrying after a few minutes.  
2. **(Step 8) Validation Failure**  
   * Required fields (e.g., test priority) are missing.  
   * *System Response:* Highlights missing fields, prompts the doctor to complete them, and prevents submission until resolved.  
3. **(Step 10) Technician Notification Failure**  
   * The system cannot reach the laboratory technicians due to network issues.  
   * *System Response:* Queues the notification for retry, logs the failure, and informs the doctor that the order is pending notification.  
4. **(Step 10) Duplicate Order Detection**  
   * The same test for the same patient has already been requested within the last 24 hours.  
   * *System Response:* Warns the doctor about the duplicate, offers to modify or cancel the new request, and prevents accidental duplication.