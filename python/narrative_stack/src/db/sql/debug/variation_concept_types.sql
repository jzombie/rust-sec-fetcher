SELECT DISTINCT ct.concept_type
FROM us_gaap_concept c
JOIN us_gaap_concept_type ct ON c.concept_type_id = ct.id
WHERE EXISTS (
    SELECT 1
    FROM us_gaap_concept_description_variation v
    WHERE v.us_gaap_concept_id = c.id
);
