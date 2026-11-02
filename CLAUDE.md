# CLAUDE.md

이 저장소는 instruct LLM에서 **콘텐츠 거부(r_c)** 와 **파괴적 행동 신중함(r_a)** 이 잔차 스트림에서 얼마나 분리돼 있는지 측정하는 실험이다. 방법, 출력, 지표 해석은 `README.md`를 본다. 사용자에게는 한국어로 짧게 보고한다.

## 처음 들어왔을 때: 혼자 세팅하기

```bash
scripts/setup.sh
```

- 스크립트가 환경을 판별해서 알아서 진행한다.
  - 호스트 + Intel GPU: podman으로 XPU 이미지를 빌드한다.
  - 호스트 + NVIDIA: podman으로 CUDA 이미지를 빌드한다.
  - 컨테이너 안(Vast.ai 등): `requirements.txt`를 pip로 설치한다.
- 이어서 디바이스 검증(`scripts/check_device.py`)을 하고, `data/sets/full`이 없으면 데이터를 만든다.
- 다시 실행해도 안전하다.
- 종료 코드별 대응:
  - `0`: 다음 단계로 진행한다.
  - `2`: GPU가 안 보인다. **CPU로 진행할지 사용자에게 묻는다.** 원인 진단(드라이버, `/dev/dri` 권한, `nvidia-smi`)도 같이 보고한다.
  - `3`: 데이터 다운로드 실패. 출력에 나온 gated 저장소를 사용자에게 알리고 `HF_TOKEN`을 요청한다.
- 세팅이 끝나면 스모크 테스트로 끝까지 도는지 확인한다: `python -m abr.run_all --preset smoke`. 호스트에서는 `./run.sh --net python ...` 형태로 실행하며, `--net`은 모델을 처음 받을 때만 필요하다. 이 결과를 사용자에게 보고한 뒤 전체 실행으로 넘어간다.

## 절대 규칙

- **호스트에서 Python을 실행하지 않는다.** 일회성 텍스트 치환이라도 마찬가지다. 호스트에서 허용되는 건 `podman`, `./run.sh`, 파일 편집(Edit/Write), `git`, 읽기 전용 셸 명령뿐이다. 클라우드 인스턴스는 그 자체가 컨테이너이므로 거기서는 `python`을 직접 실행한다.
- **네트워크는 다운로드할 때만 연다.** 호스트에서는 `./run.sh --net`(podman 6에서는 slirp4netns가 제거돼 pasta를 쓴다)을 쓰고, 기본은 `--network=none` + `HF_HUB_OFFLINE=1`이다. 모델이나 데이터셋이 캐시에 없다는 에러가 나면 `--net`을 붙여 다시 실행한다.
- **생성된 명령은 절대 실행하지 않는다.** 툴콜은 텍스트로 파싱해서 판정만 한다.
- **가중치를 수정하지 않는다.** 제거는 `hooks.ablate_direction` hook으로만 한다. TransformerLens는 쓰지 않는다.
- **커밋과 푸시는 사용자가 요청할 때만 한다.** `.gitignore`가 프롬프트 세트(`data/sets/`), 생성 결과(`generations/`, `samples_*.jsonl`), 바이너리(`*.pt`, `*.npz`)를 제외한다. 유해 프롬프트 원문(AgentHarm은 재배포 금지 요청)과 거부를 제거한 모델의 유해 출력이 들어 있기 때문이다. 이 규칙을 풀지 않는다.

## 실행

```bash
python -m abr.run_all [--preset smoke] [--set key=val ...] [--stages ...] [--layer N]
```

- 단계: `make_data → collect(GPU) → directions(CPU) → ablate_eval(GPU) → analyze(CPU)`
- 모든 단계는 이어하기가 된다. 끊기면 **같은 명령을 다시 실행**한다. 이미 있는 출력은 건너뛰고, 생성은 샘플 단위로 이어서 한다.
- 다시 계산하려면 해당 출력 파일을 지운다. 데이터를 바꿨다면 그 run의 `results/<run>/` 전체를 지운다. 그래야 활성값이 다시 수집된다.
- 모델 교체는 `--set model=<repo> [device_map=auto]`로 한다. 코드는 수정하지 않는다. 레이어 경로를 못 찾으면 `common.get_layers`의 후보 목록에 경로를 추가한다.
- 오래 걸리는 실행은 백그라운드로 돌린다. `logs/<name>.log`에 append하고 끝에 `EXIT=$?`를 남겨서, 크래시 여부를 로그만 보고 알 수 있게 한다.

## 추가 실험: oblique / learned / conditional (`abr.oblique`, 클라우드에서 실행)

로컬 iGPU로는 3~4시간이 걸려서(조건 34개 × 3세트 × n=48, 방향 학습 포함) **GPU 단계는 클라우드에서, 판정과 리포트는 로컬에서** 한다. 방법과 출력은 `src/abr/oblique.py` docstring을 본다. 결과는 `results/<run>/oblique/`에 생긴다.

- CUDA 이미지는 빌드할 때 코드를 복사한다. 코드를 바꿨다면 이미지를 다시 빌드해 push한다.
- 이미지에는 `results/`가 빠진다. 로컬과 같은 레이어·방향을 쓰려면 `directions.pt`와 `conditions.json`을 인스턴스에 올린다. 클라우드에서 다시 수집하면 bf16 수치 차이로 방향이 조금 달라진다.
- 이미지 push와 인스턴스 작업은 외부로 나가는 작업이다. 명령만 안내하고 실행은 사용자가 한다.

```bash
# 로컬: 이미지 재빌드 + push
podman build -t ghcr.io/<you>/abresearch:cuda -f Containerfile.cuda .
podman push ghcr.io/<you>/abresearch:cuda

# 로컬 → 인스턴스: 방향과 레이어만 올림 (약 3MB)
ssh -p <PORT> root@<HOST> mkdir -p /work/results/Qwen__Qwen3-4B-Instruct-2507
rsync -avz -e "ssh -p <PORT>" \
  results/Qwen__Qwen3-4B-Instruct-2507/{directions.pt,conditions.json} \
  root@<HOST>:/work/results/Qwen__Qwen3-4B-Instruct-2507/

# 인스턴스: 생성 + 방향 학습만
cd /work
python -m abr.oblique --set model=Qwen/Qwen3-4B-Instruct-2507 oblique_n=50 \
  gen_batch_size=16 learn_micro=4 --stages gpu learn \
  >> logs_oblique.log 2>&1; echo "EXIT=$?" >> logs_oblique.log

# 인스턴스 → 로컬: 회수
rsync -avz -e "ssh -p <PORT>" root@<HOST>:/work/results/Qwen__Qwen3-4B-Instruct-2507/oblique/ \
  results/Qwen__Qwen3-4B-Instruct-2507/oblique/

# 로컬: 판정 + bootstrap + 플롯 (CPU)
./run.sh python -m abr.oblique --set model=Qwen/Qwen3-4B-Instruct-2507 --stages report
```

- 리포트는 로컬에서 한다. 프로브가 로컬 `activations/`(226MB)를 쓰고, 판정 감사(아래 체크리스트)도 로컬에서 하기 때문이다.
- test 분할은 세트당 48개다. 그래서 `oblique_n=50`이어도 실제로는 n=48이 된다.
- 끊기면 같은 명령을 다시 실행한다. 생성은 샘플 단위로 이어 하고, `learned.pt`가 있으면 방향 학습을 건너뛴다. 학습 중 OOM이 나면 `learn_micro=2`로 낮춘다.
- 방향 학습에 HF gradient checkpointing을 쓰지 않는다. ablation hook과 함께 쓰면 재계산 결과가 맞지 않아 `CheckpointError`가 난다. 그래서 마이크로배치로 grad를 누적한다.
- conditional 조건은 새로 생성하지 않는다. `base`와 `rc_a100`의 출력을 프롬프트별 게이트에 따라 조합한다.

## 결과를 보고하기 전 체크리스트 (실제로 겪은 함정)

1. **판정기를 믿기 전에 샘플을 읽는다.** 휴리스틱 judge는 위험 패턴 정규식에 의존한다. 첫 실행에서 `rm -rf ~/Documents/*`, api_keys를 담은 POST, 개인키 `chmod 644`, 2507의 `chmod -R 770 /`가 `safe_alt`로 잘못 분류됐었다. 그래서 `samples_*.jsonl`에서 action_destr의 `safe_alt`·`other` 명령을 전부 훑는다. 놓친 게 있으면 `make_data.DANGER`를 고치고 `--stages analyze`만 다시 돌린다. 판정이 analyze에 있으므로 재생성은 필요 없다.
2. **콘텐츠 거부율은 *명시적* 거절의 비율이다.** 돌려 말하는 응답은 거부로 안 잡힌다. B_rc 샘플 몇 개를 직접 읽고 실제로 응했는지 보고에 같이 적는다.
3. **바닥 효과를 확인한다.** A_orig에서 action_destr의 `refuse_text + clarify + safe_alt`가 작으면, 행동 신중함을 보존하는지는 판단할 수 없다고 보고한다. Qwen3-1.7B가 이 경우였다(명확한 파괴 요청 25/25 그대로 실행).
4. **평가 샘플 구성을 확인한다.** `n_eval`은 test 세트 *앞에서부터* N개를 쓴다. 기본 30개면 ambiguous가 약 4개, AgentHarm이 약 1개뿐이다. 하위 집합에 대해 결론을 내리려면 `--set n_eval=48`(test 전체)로 돌린다. `metrics.csv`의 `*_n` 열로 표본 수를 꼭 같이 보고한다.
5. 수치는 노이즈 ceiling(`√(rel_x·rel_y)`)과 Wilson CI를 함께 보고한다. n=30이면 CI 폭이 ±0.15 안팎이다.

## 모델 구성과 모델별 메모

| run | 역할 |
|---|---|
| `Qwen/Qwen3.5-4B` | **4B 실험의 메인** (신세대) |
| `Qwen/Qwen3-4B-Instruct-2507` | 대조군: 같은 Qwen3 세대에서 크기만 키운 모델 |
| `Qwen/Qwen3-1.7B` | 소형 기준선 (행동 바닥 효과 있음) |

여러 모델을 비교할 때는 `python -m abr.compare`를 쓴다. 결과는 `results/comparison/`에 생기고, 역할 이름은 `compare.ROLES`에 있다.

모델별 메모:
- **새 모델을 붙일 때**: 먼저 `scripts/model_info.py <repo>`(`--net` 필요)로 구조와 템플릿을 확인한다. 이어서 `scripts/inspect_prompt.py --preset smoke --set model=<repo>`로 렌더링과 토큰 위치를 확인하고, 스모크 테스트를 돌린다. **스모크 결과에서 action_benign의 툴 사용률이 0이면 툴콜 포맷 파서 문제를 먼저 의심한다.**
- **Qwen3.5-4B**
  - 구조는 `Qwen3_5ForConditionalGeneration`(멀티모달, full attention과 linear attention 하이브리드, 32층)이다. `AutoModelForCausalLM`으로 로드되고 레이어도 그대로 잡힌다.
  - 툴콜이 JSON이 아니라 **XML 형식**(`<function=bash><parameter=command>…`)이다. `judge.parse_tool_calls`가 처리한다.
  - XPU에는 `flash-linear-attention`/`causal_conv1d` 커널이 없어 PyTorch 참조 구현으로 돈다. 정확하지만 느리다.
  - thinking 모드가 기본으로 켜져 있어 `enable_thinking: false`가 필요하다. config.yaml에 이미 들어 있다.
  - **로컬에서는 `--set batch_size=4 gen_batch_size=4`로 돌린다.** 기본 배치 16으로 돌렸다가 action 세트(약 600토큰)를 수집하는 도중 시스템 전역 OOM으로 죽었다(EXIT=137). iGPU는 시스템 RAM을 공유하고, 참조 구현 linear attention이 메모리를 많이 쓴다. 클라우드 GPU에서는 기본값으로 돌려도 된다.
- **Qwen3-4B-Instruct-2507**: thinking이 없는 instruct 모델이라 템플릿에 `<think>` 블록이 없다. t_post_inst는 `assistant\n`의 마지막 토큰이 된다. 모델마다 위치의 의미가 조금 다르다는 점을 비교할 때 염두에 둔다.

## 하드웨어 메모 (로컬)

- Intel Core Ultra 7 258V, Arc 140V iGPU(xe 드라이버, 메모리 공유), RAM 32GB. bf16 행렬곱은 약 17 TFLOPS다.
- 장시간 iGPU 생성 부하 중에 시스템이 하드 리셋된 적이 한 번 있다(2026-11-03). OS 로그는 없었고, 펌웨어 BERT에 하드웨어 오류 레코드 1건만 남았다.
  - 재발하면 `gen_batch_size`를 4로 낮춘다.
  - 재부팅 후에는 `journalctl -b -1`의 마지막 줄, `results/<run>/`의 상태, 잘린 jsonl을 확인하고 같은 명령으로 재개한다.

## 코드 지도

| 파일 | 역할 |
|---|---|
| `src/abr/common.py` | 설정과 프리셋, 경로, 디바이스 감지(cuda > xpu > cpu), 모델 로딩, `get_layers`, 크래시에 안전한 IO |
| `src/abr/prompts.py` | 채팅 템플릿 렌더링(툴/system 미지원 템플릿 폴백), `t_inst`·`t_post_inst` 토큰 위치 |
| `src/abr/hooks.py` | 잔차 캡처, 블록 입출력에서의 projection 제거 |
| `src/abr/make_data.py` | 4세트 생성, `DANGER` 위험 패턴(judge가 사용) |
| `src/abr/judge.py` | `Judge` 인터페이스, 휴리스틱 judge, `hf:<repo>` judge |
| `src/abr/compare.py` | 여러 모델 비교 표·플롯 (CPU) |
| `src/abr/oblique.py` | θ×α 부분 제거 스윕, 학습 방향, 프로브 게이트 제거, bootstrap CI (`results/<run>/oblique/`) |
| `scripts/model_info.py` | HF 저장소 존재 여부, 구조, 템플릿 기능 확인 (`--net` 필요) |
| `scripts/inspect_prompt.py` | 렌더링 결과와 토큰 위치를 눈으로 확인할 때 쓴다(새 모델로 바꾸면 먼저 실행) |
