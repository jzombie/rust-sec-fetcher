SELECT
    c.id AS us_gaap_concept_id,
    c.name AS us_gaap_concept_name,
    ct.concept_type AS concept_type,
    bt.balance AS balance_type,
    pt.period_type AS period_type,
    GROUP_CONCAT(DISTINCT st.statement_type ORDER BY st.statement_type) AS statement_types,
    GROUP_CONCAT(DISTINCT oc.ofss_category_id ORDER BY oc.ofss_category_id) AS ofss_category_ids
FROM us_gaap_concept c
JOIN us_gaap_concept_ofss_category oc
    ON c.id = oc.us_gaap_concept_id AND oc.is_manually_mapped = 0
LEFT JOIN us_gaap_concept_statement_type cst
    ON c.id = cst.us_gaap_concept_id AND cst.is_manually_mapped = 0
LEFT JOIN us_gaap_statement_type st
    ON cst.us_gaap_statement_type_id = st.id
LEFT JOIN us_gaap_balance_type bt ON c.balance_type_id = bt.id
LEFT JOIN us_gaap_period_type pt ON c.period_type_id = pt.id
LEFT JOIN us_gaap_concept_type ct ON c.concept_type_id = ct.id
GROUP BY c.id;
