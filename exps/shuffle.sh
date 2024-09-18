for num in {1..5}
do
  python main.py --dataset=ACM-DBLP-A --shuffle=imbalanced
done
for num in {1..5}
do
  python main.py --dataset=ACM-DBLP-A --shuffle=balanced
done

for num in {1..5}
do
  python main.py --dataset=ACM-DBLP-P --shuffle=imbalanced
done
for num in {1..5}
do
  python main.py --dataset=ACM-DBLP-P --shuffle=balanced
done

for num in {1..5}
do
  python main.py --dataset=cora --shuffle=imbalanced
done
for num in {1..5}
do
  python main.py --dataset=cora --shuffle=balanced
done

for num in {1..5}
do
  python main.py --dataset=foursquare-twitter --shuffle=imbalanced
done
for num in {1..5}
do
  python main.py --dataset=foursquare-twitter --shuffle=balanced
done

for num in {1..5}
do
  python main.py --dataset=phone-email --shuffle=imbalanced
done
for num in {1..5}
do
  python main.py --dataset=phone-email --shuffle=balanced
done