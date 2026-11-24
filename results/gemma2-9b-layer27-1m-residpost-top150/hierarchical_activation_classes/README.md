# Nested activation-pattern labels

The final relation-aware labels are nested rather than independent:

```text
coarse_0 -> coarse_0_sub_0, coarse_0_sub_1
coarse_1 -> coarse_1_sub_0, coarse_1_sub_1
coarse_2 -> coarse_2_sub_0, coarse_2_sub_1
```

The three bottom classes are formed from CAR parent/constituent features using document activation-pattern similarity. Each top composite feature is mapped to the bottom class of its CAR parents, and is then split into two subgroups within that bottom class. Thus the six possible fine classes retain their bottom parent class.

The output uses:

- `bottom_cluster_labels.csv` for bottom constituent features;
- `top_cluster_labels.csv` for top composite features and their CAR parent signatures;
- `joint_relation_members.csv` to list bottom and top features in the same relation class;
- `joint_relation_summary.csv` for one-row-per-class summaries.
