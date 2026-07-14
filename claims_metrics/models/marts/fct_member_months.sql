{{ config(materialized='table') }}

with month_spine as (
    {{ dbt_utils.date_spine(
        datepart="month",
        start_date="cast('2025-01-01' as date)",
        end_date="cast('2025-07-01' as date)"
    ) }}
),

member_months as (
    select
        m.member_id,
        m.region,
        ms.date_month as month_start
    from {{ ref('dim_members') }} as m
    cross join month_spine as ms
    where ms.date_month >= date_trunc('month', m.enrollment_start)
      and ms.date_month <= date_trunc('month', m.enrollment_end)
)

select
    member_id || '-' || strftime(month_start, '%Y%m') as member_month_id,
    member_id,
    region,
    month_start
from member_months
