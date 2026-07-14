{{ config(materialized='table') }}

select
    member_id,
    region,
    cast(enrollment_start as date) as enrollment_start,
    cast(enrollment_end as date) as enrollment_end
from {{ ref('stg_members_seed') }}
