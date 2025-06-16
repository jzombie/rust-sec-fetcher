WITH RECURSIVE group_hierarchy AS (
    SELECT
        g.id AS group_id,
        g.group_name,
        g.parent_group_id,
        CAST(g.group_name AS CHAR(1000)) AS full_path
    FROM ofss_group g
    WHERE g.parent_group_id IS NULL

    UNION ALL

    SELECT
        g.id,
        g.group_name,
        g.parent_group_id,
        CONCAT(gh.full_path, ' > ', g.group_name)
    FROM ofss_group g
    JOIN group_hierarchy gh ON g.parent_group_id = gh.group_id
)

SELECT
    c.id AS us_gaap_concept_id,
    c.name AS us_gaap_concept_name,
    ct.concept_type AS concept_type,
    bt.balance AS balance_type,
    pt.period_type AS period_type,
    GROUP_CONCAT(DISTINCT oc.ofss_category_id ORDER BY oc.ofss_category_id) AS ofss_category_ids,
    GROUP_CONCAT(DISTINCT ocat.category_name ORDER BY oc.ofss_category_id) AS ofss_category_names,
    GROUP_CONCAT(DISTINCT CONCAT(gh.full_path, ' > ', ocat.category_name) ORDER BY oc.ofss_category_id) AS ofss_category_hierarchies
FROM us_gaap_concept c
JOIN us_gaap_concept_ofss_category oc
    ON c.id = oc.us_gaap_concept_id AND oc.is_manually_mapped = 0
JOIN ofss_category ocat
    ON oc.ofss_category_id = ocat.id
LEFT JOIN ofss_group g ON ocat.group_id = g.id
LEFT JOIN group_hierarchy gh ON g.id = gh.group_id
LEFT JOIN us_gaap_balance_type bt ON c.balance_type_id = bt.id
LEFT JOIN us_gaap_period_type pt ON c.period_type_id = pt.id
LEFT JOIN us_gaap_concept_type ct ON c.concept_type_id = ct.id
GROUP BY c.id;
