while :
do
    python trt_test.py 
    python demo.py 

    read -n 1 -s -t 1 input

    if [[ $input = $'e' ]]; then
        break
    fi
done

echo "ESC key pressed to end script"