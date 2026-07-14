{{ config(materialized='table') }}

select
    claim_id,
    member_id,
    drug_id,
    region,
    cast(submitted_date as date) as submitted_date,
    cast(adjudicated_date as date) as adjudicated_date,
    spend,
    denied
from {{ ref('stg_claims_seed') }}
