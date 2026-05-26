import carla


HOST = "127.0.0.1"
PORT = 2000


DESTROY_PATTERNS = [
    "vehicle.*",
    "sensor.*",
    "walker.pedestrian.*",
    "controller.ai.walker",
    "static.prop.teknofest_sign*",
    "static.prop.streetsign*",
]


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()
    print("Map:", world.get_map().name)

    actors_to_destroy = []

    for pattern in DESTROY_PATTERNS:
        actors = list(world.get_actors().filter(pattern))
        print(f"{pattern}: {len(actors)}")
        actors_to_destroy.extend(actors)

    unique = {}
    for actor in actors_to_destroy:
        unique[actor.id] = actor

    actors_to_destroy = list(unique.values())

    if not actors_to_destroy:
        print("Temizlenecek runtime actor yok.")
        return

    print("Destroy edilecek actor sayısı:", len(actors_to_destroy))

    commands = [carla.command.DestroyActor(actor.id) for actor in actors_to_destroy]
    responses = client.apply_batch_sync(commands, True)

    failed = 0
    for actor, response in zip(actors_to_destroy, responses):
        if response.error:
            failed += 1
            print(f"FAILED id={actor.id} type={actor.type_id}: {response.error}")
        else:
            print(f"DESTROYED id={actor.id} type={actor.type_id}")

    print("Bitti.")
    print("Başarılı:", len(actors_to_destroy) - failed)
    print("Hatalı:", failed)


if __name__ == "__main__":
    main()
