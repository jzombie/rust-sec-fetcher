SELECT
    c.id AS us_gaap_concept_id,
    c.name AS us_gaap_concept_name,
	ct.concept_type AS concept_type,
    bt.balance AS balance_type,
    pt.period_type AS period_type
FROM us_gaap_concept c
LEFT JOIN us_gaap_concept_ofss_category m ON c.id = m.us_gaap_concept_id
LEFT JOIN us_gaap_balance_type bt ON c.balance_type_id = bt.id
LEFT JOIN us_gaap_period_type pt ON c.period_type_id = pt.id
LEFT JOIN us_gaap_concept_type ct ON c.concept_type_id = ct.id
WHERE m.ofss_category_id IS NULL;
