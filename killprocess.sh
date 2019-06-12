for pid in `ps aux | grep test.py | awk '{print $2}'`; do
    kill -9 $pid
done
