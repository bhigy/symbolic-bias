for d in $(ls -d *); do [ -f $d/result.json ] && cat $d/result.json | ./res2csv.sh > $d/result.csv; done
