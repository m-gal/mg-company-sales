""" Contains project configuration stuff

    @author: mikhail.galkin
"""

# ------------------------------------------------------------------------------
# ------------------------ D A T A S E T  S C H E M A  -------------------------
# ------------------------------------------------------------------------------
COLS_NA_VALUES = ["Unknown"]

COLS_TYPES_INT = [
    "delivery_point_check_digit",
    "company_started_year", # "company_year_started"
    "company_started_month", # "company_month_started"
    "location_started_month", # "location_month_started"
    "location_started_year", # "location_year_started"
    "num_units_in_chain",
    "num_beds",
    "number_of_pcs",
    "company_employees_exact", # "company_exact_num_emp"
    "company_sale_volume_exact", # "exact_sales_volume"
    "company_employee_num_exact", # "exact_number_of_employees"
    "company_sales_exact", # "company_exact_sales"
    "e_biz_score", # "eBizScore"
    "e_biz_decile", # "eBizDecile"
    "e_biz_percentile", # "eBizPercentile"
    "number_of_tenants", # "NumberOfTenants"
    "contact_birth_year", # "birth_year"
]

COLS_TYPES_FLOAT = [
    "location_latitude", # "latitude"
    "location_longitude", # "longitude"
    "zip_centroid_lat",
    "zip_centroid_lon", # "zip_centroid_long"
]

COLS_ZERO_TO_NA = [
    "company_sale_volume_exact", # "exact_sales_volume"
    "company_employee_num_exact", # "exact_number_of_employees"
    "company_sales_exact", # "company_exact_sales"
    "company_employees_exact", # "company_exact_num_emp"
]

COLS_DTYPE = {
    "company_name": str,
    "legal_name": str,
    "trade_name": str,
    "dba_name": str,
    "fictitious_name": str,
    "address1": str,
    "address2": str,
    "city": str,
    "state": str,
    "zip5": str,
    "zip4": str,
    "address_type_code": str,
    "mail_addr_address1": str,
    "mail_addr_address2": str,
    "mail_addr_city": str,
    "mail_addr_state": str,
    "mail_addr_zip5": str,
    "mail_addr_zip4": str,
    "mail_addr_address_type_code": str,
    "country": str,
    "state_code": str,
    "county_code": str,
    "county_name": str,
    "carrier_route": str,
    "cbsa": str,
    "cbsa_name": str,
    "dma": str,
    "dma_name": str,
    "scf": str,
    "congress_code": str,
    "delivery_point_code": str,
    "delivery_point_check_digit": "Int64",  #!
    "latitude": "float64",  # *
    "longitude": "float64",  # *
    "timezone": str,
    "timezonecode": str,
    "census_id": str,
    "census_block": str,
    "census_tract": str,
    "MedianIncomeCensusArea": str,
    "MeanHousingCensusArea": str,
    "url": str,
    "url_type": str,
    "telephone": str,
    "telephone2": str,
    "toll_free_number": str,
    "fax": str,
    "fax2": str,
    "naics": str,
    "naics_desc": str,
    "sic2code": str,
    "sic4code": str,
    "sic6code": str,
    "sic2desc": str,
    "sic4desc": str,
    "sic6desc": str,
    "sic8desc": str,
    "sic_division": str,
    "exact_sales_volume": "float64",  # *
    "sales_volume": str,
    "sales_code": str,
    "exact_number_of_employees": "float64",  # *
    "number_of_employees": str,
    "employee_code": str,
    "location_type": str,
    "parent_company": str,
    "parent_address": str,
    "parent_city": str,
    "parent_state": str,
    "parent_zip": str,
    "parent_country": str,
    "parent_phone": str,
    "public": str,
    "business_specialty": str,
    "company_year_started": "Int64",  #!
    "business_type": str,
    "state_where_entity_formed": str,
    "minority": str,
    "woman": str,
    "government": str,
    "small": str,
    "home_office": str,
    "franchise": str,
    "chain": str,
    "mailable": str,
    "phoneable": str,
    "primary_record": str,
    "streetnum": str,
    "predir": str,
    "streetname": str,
    "addrsuffix": str,
    "postdir": str,
    "unit_type": str,
    "unit_num": str,
    "mail_addr_streetnum": str,
    "mail_addr_streetname": str,
    "mail_addr_unit_num": str,
    "minority_desc_code": str,
    "fka_name": str,
    "prev_address1": str,
    "prev_address2": str,
    "prev_city": str,
    "prev_state": str,
    "prev_zip5": str,
    "prev_telephone": str,
    "unpublished_telephone": str,
    "phone_type1": str,
    "phone_type2": str,
    "phone_delete_flag": str,
    "fortune_company": str,
    "foreign_company": str,
    "ein": str,
    "num_units_in_chain": "Int64",  #!
    "store_number": str,
    "company_month_started": "Int64",  #!
    "location_month_started": "Int64",  #!
    "location_year_started": "Int64",  #!
    "num_beds": "Int64",  #!
    "medicare_provider_code": str,
    "square_footage": str,
    "number_of_pcs": "Int64",  #!
    "site_status": str,
    "subsidiary_ind": str,
    "manufacturing_ind": str,
    "zip_centroid_lat": "float64",  # *
    "zip_centroid_long": "float64",  # *
    "residence_address": str,
    "company_exact_sales": "float64",  # *
    "company_sales": str,
    "company_sales_code": str,
    "company_exact_num_emp": "Int64",  #!
    "company_num_emp": str,
    "company_emp_code": str,
    "owner_company": str,
    "owner_address": str,
    "owner_city": str,
    "owner_state": str,
    "owner_zip": str,
    "owner_country": str,
    "owner_phone": str,
    "ra_name": str,
    "ra_address": str,
    "ra_city": str,
    "ra_state": str,
    "ra_zip": str,
    "ra_country": str,
    "ra_phone": str,
    "eBizScore": "Int64",  #!
    "eBizDecile": "Int64",  #!
    "eBizPercentile": "Int64",  #!
    "MonthlyTelcoSpend": "Int64",  #!
    "MonthlyDataSpend": "Int64",  #!
    "MTUID": str,
    "NumberOfTenants": "Int64",  #!
    "Lines": str,
    "contact_name": str,
    "prefix": str,
    "first_name": str,
    "middle_name": str,
    "surname": str,
    "suffix": str,
    "ethnicity": str,
    "gender": str,
    "contact_address1": str,
    "contact_address2": str,
    "contact_city": str,
    "contact_state": str,
    "contact_zip5": str,
    "contact_zip4": str,
    "contact_telephone": str,
    "orig_title1": str,
    "orig_title2": str,
    "birth_year": "Int64",  #!
}

COLS_RENAME = {
    # company name -------------------------------------------------------------
    "legal_name": "company_name_legal",
    "trade_name": "company_name_trade",
    "dba_name": "company_name_dba",
    "fictitious_name": "company_name_fictitious",
    "fka_name": "company_name_fka",
    # address ------------------------------------------------------------------
    "address1": "addr_address_1",
    "address2": "addr_address_2",
    "address_type_code": "addr_address_type_code",
    "city": "addr_city",
    "state": "addr_state",
    "zip4": "addr_zip_4",
    "zip5": "addr_zip_5",
    "state_code": "addr_state_code",
    "country": "addr_country",
    "county_code": "addr_county_code",
    "county_name": "addr_county_name",
    "streetnum": "addr_street_num",
    "streetname": "addr_street_name",
    "addrsuffix": "addr_suffix",
    "unit_num": "addr_unit_num",
    "unit_type": "addr_unit_type",
    # codes --------------------------------------------------------------------
    "cbsa": "cbsa_code",
    "dma": "dma_code",
    "scf": "scf_code",
    "naics": "naics_code",
    "ein": "ein_code",
    "timezonecode": "timezone_code",
    # mails --------------------------------------------------------------------
    "mail_addr_streetnum": "mail_addr_street_num",
    "mail_addr_streetname": "mail_addr_street_name",
    # location -----------------------------------------------------------------
    "latitude": "location_latitude",
    "longitude": "location_longitude",
    "zip_centroid_long": "zip_centroid_lon",
    # indicators ---------------------------------------------------------------
    "exact_sales_volume": "company_sale_volume_exact",
    "sales_volume": "company_sale_volume",
    "sales_code": "company_sale_code",
    "exact_number_of_employees": "company_employee_num_exact",
    "number_of_employees": "company_employee_num",
    "employee_code": "company_employee_code",
    "company_exact_sales": "company_sales_exact",
    "company_sales": "company_sales",
    "company_sales_code": "company_sales_code",
    "company_exact_num_emp": "company_employees_exact",
    "company_num_emp": "company_employees",
    "company_emp_code": "company_employees_code",
    # sic ----------------------------------------------------------------------
    "sic2code": "sic_2_code",
    "sic2desc": "sic_2_desc",
    "sic4code": "sic_4_code",
    "sic4desc": "sic_4_desc",
    "sic6code": "sic_6_code",
    "sic6desc": "sic_6_desc",
    "sic8desc": "sic_8_desc",
    # contact ------------------------------------------------------------------
    "prefix": "contact_prefix",
    "suffix": "contact_suffix",
    "first_name": "contact_name_first",
    "middle_name": "contact_name_middle",
    "surname": "contact_name_surname",
    "ethnicity": "contact_ethnicity",
    "gender": "contact_gender",
    "birth_year": "contact_birth_year",
    "orig_title1": "contact_orig_title_1",
    "orig_title2": "contact_orig_title_2",
    # misc ---------------------------------------------------------------------
    "MedianIncomeCensusArea": "census_median_income_area",
    "MeanHousingCensusArea": "census_mean_housing_area",
    "company_month_started": "company_started_month",
    "company_year_started": "company_started_year",
    "location_month_started": "location_started_month",
    "location_year_started": "location_started_year",
}

COLS_DROP = ["Lines", "Unnamed: 173"]
