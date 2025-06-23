SELECT
  SUM(CASE WHEN m.us_gaap_concept_id IS NULL THEN 1 ELSE 0 END) AS unmapped_count,
  SUM(CASE WHEN m.us_gaap_concept_id IS NOT NULL THEN 1 ELSE 0 END) AS mapped_count
FROM us_gaap_concept c
LEFT JOIN us_gaap_concept_ofss_category m
  ON c.id = m.us_gaap_concept_id;
