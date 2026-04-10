

import csv
import requests
from datetime import datetime

# Define the API endpoint
url = 'https://api-trestle.corelogic.com/trestle/odata/Property'

# Get token from IDX Exchange secure proxy instead of exposing CoreLogic credentials
auth_endpoint = 'https://idxexchange.com/internal-api/trestle_token.php?key=IDXEXCHANGE2026_CHANGE_THIS'

response = requests.get(auth_endpoint, timeout=30)
response.raise_for_status()



# Parse the response to extract the token
token = response.json().get('access_token')

if token:

    # Define the headers with the token
    headers = {
        'Authorization': f'Bearer {token}'
    }

    # Define the parameters for the API request
    params = {
        '$select': 'OriginalListPrice, ListingKey,CloseDate,ClosePrice,ListAgentFirstName,ListAgentLastName,Latitude,Longitude,UnparsedAddress,PropertyType,LivingArea,ListPrice,DaysOnMarket,ListOfficeName,BuyerOfficeName,CoListOfficeName,ListAgentFullName,CoListAgentFirstName,CoListAgentLastName,BuyerAgentMlsId,BuyerAgentFirstName,BuyerAgentLastName,FireplacesTotal,AssociationFeeFrequency,AboveGradeFinishedArea,ListingKeyNumeric,MLSAreaMajor,TaxAnnualAmount,CountyOrParish,PropertyType,MlsStatus,ElementarySchool,ListAgentFirstName,AttachedGarageYN,ParkingTotal,BuilderName,PropertySubType,LotSizeAcres,SubdivisionName,BuyerOfficeAOR,YearBuilt,DaysOnMarket,StreetNumberNumeric,LivingArea,ListingId,BathroomsTotalInteger,City,TaxYear,BuildingAreaTotal,BedroomsTotal,ContractStatusChangeDate,Longitude,ElementarySchoolDistrict,CoBuyerAgentFirstName,PurchaseContractDate,ListingContractDate,BelowGradeFinishedArea,BusinessType,Latitude,ListPrice,StateOrProvince,CoveredSpaces,MiddleOrJuniorSchool,FireplaceYN,Stories,HighSchool,Levels,ListAgentLastName,CloseDate,LotSizeDimensions,LotSizeArea,MainLevelBedrooms,NewConstructionYN,GarageSpaces,HighSchoolDistrict,PostalCode,BuyerOfficeName,AssociationFee,LotSizeSquareFeet,MiddleOrJuniorSchoolDistrict,UnparsedAddress',
      
        '$filter': f"ListingContractDate ge {datetime(2026, 3, 1).isoformat(timespec='milliseconds')}Z and ListingContractDate lt {datetime(2026, 4, 1).isoformat(timespec='milliseconds')}Z",


        '$top': 1000  # Extracting up to 1000 observations
    }

    # Send a GET request to the API endpoint with the token and parameters
    total_records = 0
    csv_file = 'data/CRMLSListing202603.csv'

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['OriginalListPrice','ListingKey', 'CloseDate', 'ClosePrice', 'ListAgentFirstName', 'ListAgentLastName', 'Latitude', 'Longitude', 'UnparsedAddress', 'PropertyType', 'LivingArea', 'ListPrice', 'DaysOnMarket', 'ListOfficeName', 'BuyerOfficeName', 'CoListOfficeName', 'ListAgentFullName', 'CoListAgentFirstName', 'CoListAgentLastName', 'BuyerAgentMlsId', 'BuyerAgentFirstName', 'BuyerAgentLastName', 'FireplacesTotal', 'AssociationFeeFrequency', 'AboveGradeFinishedArea', 'ListingKeyNumeric', 'MLSAreaMajor', 'TaxAnnualAmount', 'CountyOrParish', 'PropertyType', 'MlsStatus', 'ElementarySchool', 'ListAgentFirstName', 'AttachedGarageYN', 'ParkingTotal', 'BuilderName', 'PropertySubType', 'LotSizeAcres', 'SubdivisionName', 'BuyerOfficeAOR', 'YearBuilt', 'DaysOnMarket', 'StreetNumberNumeric', 'LivingArea', 'ListingId', 'BathroomsTotalInteger', 'City',  'TaxYear', 'BuildingAreaTotal', 'BedroomsTotal', 'ContractStatusChangeDate', 'Longitude', 'ElementarySchoolDistrict', 'CoBuyerAgentFirstName', 'PurchaseContractDate', 'ListingContractDate', 'BelowGradeFinishedArea', 'BusinessType', 'Latitude', 'ListPrice', 'StateOrProvince', 'CoveredSpaces', 'MiddleOrJuniorSchool', 'FireplaceYN', 'Stories', 'HighSchool', 'Levels', 'ListAgentLastName', 'CloseDate', 'LotSizeDimensions', 'LotSizeArea', 'MainLevelBedrooms', 'NewConstructionYN', 'GarageSpaces', 'HighSchoolDistrict', 'PostalCode', 'BuyerOfficeName', 'AssociationFee', 'LotSizeSquareFeet', 'MiddleOrJuniorSchoolDistrict', 'UnparsedAddress'])
        writer.writeheader()

        while True:
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                observations = data.get('value', [])
                for observation in observations:
                    writer.writerow({
                      'OriginalListPrice': observation.get('OriginalListPrice', ''),
                        'ListingKey': observation.get('ListingKey', ''),            
                        'CloseDate': observation.get('CloseDate', ''),
                        'ClosePrice': observation.get('ClosePrice', ''),
                        'ListAgentFirstName': observation.get('ListAgentFirstName', ''),
                        'ListAgentLastName': observation.get('ListAgentLastName', ''),
                        'Latitude': observation.get('Latitude', ''),
                        'Longitude': observation.get('Longitude', ''),
                        'UnparsedAddress': observation.get('UnparsedAddress', ''),
                        'PropertyType': observation.get('PropertyType', ''),
                        'LivingArea': observation.get('LivingArea', ''),
                        'ListPrice': observation.get('ListPrice', ''),
                        'DaysOnMarket': observation.get('DaysOnMarket', ''),
                        'ListOfficeName': observation.get('ListOfficeName', ''),
                        'BuyerOfficeName': observation.get('BuyerOfficeName', ''),
                        'CoListOfficeName': observation.get('CoListOfficeName', ''),
                        'ListAgentFullName': observation.get('ListAgentFullName', ''),
                        'CoListAgentFirstName': observation.get('CoListAgentFirstName', ''),
                        'CoListAgentLastName': observation.get('CoListAgentLastName', ''),
                        'BuyerAgentMlsId': observation.get('BuyerAgentMlsId', ''),
                        'BuyerAgentFirstName': observation.get('BuyerAgentFirstName', ''),
                        'BuyerAgentLastName': observation.get('BuyerAgentLastName', ''),
                        'FireplacesTotal': observation.get('FireplacesTotal', ''),
                        'AssociationFeeFrequency': observation.get('AssociationFeeFrequency', ''),
                        'AboveGradeFinishedArea': observation.get('AboveGradeFinishedArea', ''),
                        'ListingKeyNumeric': observation.get('ListingKeyNumeric', ''),
                        'MLSAreaMajor': observation.get('MLSAreaMajor', ''),
                        'TaxAnnualAmount': observation.get('TaxAnnualAmount', ''),
                        'CountyOrParish': observation.get('CountyOrParish', ''),
                        'MlsStatus': observation.get('MlsStatus', ''),
                        'ElementarySchool': observation.get('ElementarySchool', ''),
                        'ListAgentFirstName': observation.get('ListAgentFirstName', ''),
                        'AttachedGarageYN': observation.get('AttachedGarageYN', ''),
                        'ParkingTotal': observation.get('ParkingTotal', ''),
                        'BuilderName': observation.get('BuilderName', ''),
                        'PropertySubType': observation.get('PropertySubType', ''),
                        'LotSizeAcres': observation.get('LotSizeAcres', ''),
                        'SubdivisionName': observation.get('SubdivisionName', ''),
                        'BuyerOfficeAOR': observation.get('BuyerOfficeAOR', ''),
                        'YearBuilt': observation.get('YearBuilt', ''),
                        'DaysOnMarket': observation.get('DaysOnMarket', ''),                        
                        'StreetNumberNumeric': observation.get('StreetNumberNumeric', ''),
                        'LivingArea': observation.get('LivingArea', ''),
                        'ListingId': observation.get('ListingId', ''),
                        'BathroomsTotalInteger': observation.get('BathroomsTotalInteger', ''),
                        'City': observation.get('City', ''),
                        'TaxYear': observation.get('TaxYear', ''),
                        'BuildingAreaTotal': observation.get('BuildingAreaTotal', ''),
                        'BedroomsTotal': observation.get('BedroomsTotal', ''),
                        'ContractStatusChangeDate': observation.get('ContractStatusChangeDate', ''),
                        'Longitude': observation.get('Longitude', ''),
                        'ElementarySchoolDistrict': observation.get('ElementarySchoolDistrict', ''),
                        'CoBuyerAgentFirstName': observation.get('CoBuyerAgentFirstName', ''),
                        'PurchaseContractDate': observation.get('PurchaseContractDate', ''),
                        'ListingContractDate': observation.get('ListingContractDate', ''),
                        'BelowGradeFinishedArea': observation.get('BelowGradeFinishedArea', ''),
                        'BusinessType': observation.get('BusinessType', ''),
                        'Latitude': observation.get('Latitude', ''),
                        'ListPrice': observation.get('ListPrice', ''),
                        'StateOrProvince': observation.get('StateOrProvince', ''),
                        'CoveredSpaces': observation.get('CoveredSpaces', ''),
                        'MiddleOrJuniorSchool': observation.get('MiddleOrJuniorSchool', ''),
                        'FireplaceYN': observation.get('FireplaceYN', ''),
                        'Stories': observation.get('Stories', ''),
                        'HighSchool': observation.get('HighSchool', ''),
                        'Levels': observation.get('Levels', ''),
                        'ListAgentLastName': observation.get('ListAgentLastName', ''),
                        'CloseDate': observation.get('CloseDate', ''),
                        'LotSizeDimensions': observation.get('LotSizeDimensions', ''),
                        'LotSizeArea': observation.get('LotSizeArea', ''),
                        'MainLevelBedrooms': observation.get('MainLevelBedrooms', ''),
                        'NewConstructionYN': observation.get('NewConstructionYN', ''),
                        'GarageSpaces': observation.get('GarageSpaces', ''),
                        'HighSchoolDistrict': observation.get('HighSchoolDistrict', ''),
                        'PostalCode': observation.get('PostalCode', ''),
                        'BuyerOfficeName': observation.get('BuyerOfficeName', ''),
                        'AssociationFee': observation.get('AssociationFee', ''),
                        'LotSizeSquareFeet': observation.get('LotSizeSquareFeet', ''),
                        'MiddleOrJuniorSchoolDistrict': observation.get('MiddleOrJuniorSchoolDistrict', ''),
                        'UnparsedAddress': observation.get('UnparsedAddress', '')
                    })
                    total_records += 1

                # Check if there are more records to fetch
                if '@odata.nextLink' in data:
                    next_link = data['@odata.nextLink']
                    params = None  # Clear params to avoid appending to the existing query string
                    url = next_link
                else:
                    break
            else:
                print(f"Error: {response.status_code}")
                print(f"Error Message: {response.text}")
                break

    print(f"Total {total_records} records exported to {csv_file}")
else:
    print("Error retrieving token: access_token not found")
