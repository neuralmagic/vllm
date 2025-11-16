for f in *-of-00272.safetensors; do
    [ -e "$f" ] || continue
    new="${f/-of-00272.safetensors/-of-00273.safetensors}"
    mv "$f" "$new"
done

