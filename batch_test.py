from quick_test import quick_test

MODES = ["baseline", "ia3"]


def run_all():
    results = {m: quick_test(m) for m in MODES}
    for m, res in results.items():
        print(f"{m}: {'OK' if res else 'FAIL'}")
    return all(results.values())


if __name__ == "__main__":
    success = run_all()
    raise SystemExit(0 if success else 1)
