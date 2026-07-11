import argparse

from paper_agent.memory import UserProfileStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Show or update long-term user profile memory")
    parser.add_argument("--memory", default="artifacts/memory/user_profile.json")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"))
    args = parser.parse_args()

    store = UserProfileStore(args.memory)
    if args.set:
        key, value = args.set
        profile = store.set(key, value)
        print(f"updated: {key}")
    else:
        profile = store.load()
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
