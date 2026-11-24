# Relation-aware nested feature classes

These files keep the CAR bottom-to-top mapping while adding activation-pattern labels.

- `bottom_cluster_labels.csv`: bottom constituent features and their 3 coarse classes.
- `top_cluster_labels.csv`: top composite features, the bottom parent-cluster signature, and the nested top subgroup.
- `joint_relation_members.csv`: bottom and top features listed together under one `class_id`.
- `joint_relation_summary.csv`: one row per joint relation class.

Use `class_id` from `joint_relation_members.csv` when selecting a complete group for downstream data generation. The `role` column distinguishes `bottom_constituent` from `top_composite`.
