### **Results for attached Sample EML Files**
| EML File Name | Email Content Summary | Expected Request Type(s) | Confidence Score(s) | Extracted Key Fields | Duplicate Detected |
|--------------|----------------------|-------------------|----------------|------------------|-----------------|
| loan_request_001.eml | Customer requesting a new mortgage loan | New Loan Request | 0.92 | Loan Amount, Term, Interest Rate | No |
| loan_status_002.eml | Inquiry about the status of an existing loan | Loan Status Inquiry | 0.88 | Loan ID, Current Status | No |
| payment_issue_003.eml | Customer reporting payment processing issues | Payment Issue | 0.85 | Transaction ID, Amount | No |
| multiple_requests_004.eml | Customer requesting loan modification and account balance | Loan Modification, Balance Inquiry | 0.78, 0.82 | Loan ID, New Terms, Account Balance | No |
| duplicate_loan_request_005.eml | Similar to loan_request_001.eml but sent again | New Loan Request | 0.92 | Loan Amount, Term, Interest Rate | Yes |
