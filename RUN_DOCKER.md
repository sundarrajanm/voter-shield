To load the docker image run the following command:

```bash
docker load -i voter-shield-app
```

To run the docker image run the following command:

```bash
docker-compose run --rm app python main.py --s3-input https://264676382451-eci-download.s3.ap-southeast-2.amazonaws.com/sample/sample_tamil.pdf ----output-identifier 53_ts_2026_f1 --s3-output s3://264676382451-eci-download/voter-shield/53_ts_2026_f1
```
