# EDA Findings (Day 1)

## Record counts

- S1 train: 2,206,821
- S2 train: 5,034,616
- S3 train: 5,285,603
- ground truth rows: 2,206,821

## Q3: one-to-one assignment

- distinct matched S2/S3 ids: 7,638,365
- ids matched to MORE than one S1: 0 (0.0000%)

## Q5: country consistency within matched pairs

- total ground-truth pairs: 7,638,365
- pairs where S1 country != matched-record country: 0 (0.0000%)

## Q4: orphan rate (records matching no S1 at all)

- S2 orphans: 1,340,997 / 5,034,616 (26.6355%)
- S3 orphans: 1,340,857 / 5,285,603 (25.3681%)

## Q10: postal-code-like token presence (rough regex, per source)

- S1: 6.6728%
- S2: 7.3320%
- S3: 7.2977%

## Q12: non-ASCII character presence in business_name (by source)

- S1: 0.0000% of names contain non-ASCII characters
- S2: 15.1870% of names contain non-ASCII characters
- S3: 11.4790% of names contain non-ASCII characters

## Country mix (train)

- S1: {'US': 1323633, 'India': 883188}
- S2: {'India': 2017799, 'US': 3016817}
- S3: {'US': 3170056, 'India': 2115547}
