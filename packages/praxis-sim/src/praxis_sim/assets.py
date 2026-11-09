"""Download and check the open-source robot models of the simulation.

The robot models come from MuJoCo Menagerie
(https://github.com/google-deepmind/mujoco_menagerie) at a pinned commit:

* ``franka_emika_panda/panda_nohand.xml``: the Franka Emika Panda arm without
  its hand. License: Apache-2.0. Menagerie made it from the public
  ``franka_description`` package of ``franka_ros``.
* ``robotiq_2f85/2f85.xml``: the Robotiq 2F-85 gripper. License:
  BSD-2-Clause, "Copyright (c) 2013, ROS-Industrial". Menagerie made it from
  the public ROS-Industrial ``robotiq`` package.
* ``i2rt_yam/yam_linear_4310.xml``: the I2RT YAM arm with the ``linear_4310``
  parallel gripper (the arm of the bimanual rig). License: MIT, "Copyright
  (c) 2025 i2rt robotics". Menagerie made it from the public I2RT URDF and
  SDK files.

This repository does not contain the model files. :func:`ensure_assets`
downloads each file from the pinned commit into a cache directory and checks
its SHA-256. The LICENSE file of each model directory is in the download.
The cache directory is ``$PRAXIS_SIM_CACHE``, or
``$XDG_CACHE_HOME/praxis/sim``, or ``~/.cache/praxis/sim``.

To use a local Menagerie copy (for example on a machine without a network),
set ``source`` to its directory. The function then copies the files and
checks the same SHA-256 values.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
RAW_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/{path}"
)
DOWNLOAD_TIMEOUT_S = 30.0
DOWNLOAD_WORKERS = 8


@dataclass(frozen=True)
class ModelSource:
    """Source and license of one robot model."""

    name: str
    directory: str
    xml: str
    license: str
    copyright: str
    upstream: str


MODELS: dict[str, ModelSource] = {
    "panda": ModelSource(
        name="Franka Emika Panda (no hand)",
        directory="franka_emika_panda",
        xml="franka_emika_panda/panda_nohand.xml",
        license="Apache-2.0",
        copyright="MuJoCo Menagerie (Google DeepMind), from Franka Emika files",
        upstream="https://github.com/frankaemika/franka_ros",
    ),
    "robotiq_2f85": ModelSource(
        name="Robotiq 2F-85",
        directory="robotiq_2f85",
        xml="robotiq_2f85/2f85.xml",
        license="BSD-2-Clause",
        copyright="Copyright (c) 2013, ROS-Industrial",
        upstream="https://github.com/ros-industrial/robotiq",
    ),
    "yam": ModelSource(
        name="I2RT YAM with the linear_4310 gripper",
        directory="i2rt_yam",
        xml="i2rt_yam/yam_linear_4310.xml",
        license="MIT",
        copyright="Copyright (c) 2025 i2rt robotics",
        upstream="https://github.com/i2rt-robotics/i2rt",
    ),
}

# (path in the Menagerie repository, sha256, size in bytes) at MENAGERIE_COMMIT.
FILES: tuple[tuple[str, str, int], ...] = (
    (
        "i2rt_yam/LICENSE",
        "dc46c3a7dc48221aa783e634d15db0f816ba83ba5cdb77bbaa91525c61c7552f",
        1070,
    ),
    (
        "i2rt_yam/assets/linear_4310/d405_wrist_linear_4310.stl",
        "f01bcd92b447c3a44c1ad9345367628d4157fe4c07b545ccbf9bc4ae81115112",
        469584,
    ),
    (
        "i2rt_yam/assets/linear_4310/gripper.stl",
        "d949c401186197d842ca8f9b23bfcbf24bed8461b421df06093c65e4ead21f5d",
        400084,
    ),
    (
        "i2rt_yam/assets/linear_4310/tip_left.stl",
        "03d94f53d7b5e9e1871ea6f3925c68d7a2f894b08e0c575fa0b9ed4024684ff7",
        400084,
    ),
    (
        "i2rt_yam/assets/linear_4310/tip_right.stl",
        "8f620aae9c15e00e2ea4e5bf5e9a49e7edb2d23128b8955cfacc0da8da45c72f",
        400084,
    ),
    (
        "i2rt_yam/assets/model2.stl",
        "5ac61c0313ed00c38655e41681af1b8b307cf8d5ef57209a75f215cea02df754",
        524284,
    ),
    (
        "i2rt_yam/assets/model2__10.stl",
        "b879cc07f547b7a2ac52e428dd206e80a0d4b0e805cb56b9a8230df04a690923",
        524284,
    ),
    (
        "i2rt_yam/assets/model2__11.stl",
        "af220d2a51e8444392aef36e2fe4a6f69aed22bc292a7f38e54da3b4f5253434",
        248584,
    ),
    (
        "i2rt_yam/assets/model2__2.stl",
        "30dab41f14bae95d737abca3c934eb5d570ab107dff0b40952fbdfb9d07eab79",
        524284,
    ),
    (
        "i2rt_yam/assets/model2__3.stl",
        "9fd96f5ad387fe7dd96eaa65e386637800e2f5637b6167fbad4ecb37d8598f47",
        80684,
    ),
    (
        "i2rt_yam/assets/model2__4.stl",
        "33c0e7ef538e2c4473efa9c4ef2a8441b6094b65090b4dec71abe82001d974df",
        80684,
    ),
    (
        "i2rt_yam/assets/model2__5.stl",
        "09f08e7dead2d6363618471adadb1b94af39e8056676670f3fd21ec06c197b20",
        524284,
    ),
    (
        "i2rt_yam/assets/model2__6.stl",
        "888b5919d10dbfb59b964c3cafc2ddbdac3b5426e20dd13bcaf4baedd5ee15c9",
        86284,
    ),
    (
        "i2rt_yam/assets/model2__7.stl",
        "2d64b18f1e70b0460eb53335d340b7c879976a3093ec67b07bcb7245e371a91d",
        52084,
    ),
    (
        "i2rt_yam/assets/model2__8.stl",
        "db3708da6b14daf3fc9a1909a8325b68f65ff69146cdd20b69a9ddc4ce51f50d",
        524284,
    ),
    (
        "i2rt_yam/assets/model2__9.stl",
        "74969d09b8b7f75cc984c5c7e5526d2941c6724f48b3efa1fa8730cc7820f4f5",
        52084,
    ),
    (
        "i2rt_yam/yam_linear_4310.xml",
        "beb1ccea8eefc3e181a562459fb55626e4beb7ef63eb0ceac55bdf8f24c63434",
        16991,
    ),
    (
        "franka_emika_panda/LICENSE",
        "a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9",
        10173,
    ),
    (
        "franka_emika_panda/assets/link0.stl",
        "dfc6d94330de8ddb005b311bfdba9f3b8e1aa7c256b71592ee7ff32cb9a9a5aa",
        10084,
    ),
    (
        "franka_emika_panda/assets/link0_0.obj",
        "6b9f41a4c203540daaeb7a33cc3d0d6c6a32dc8bba187074eeaf97b3e36b3f9c",
        296497,
    ),
    (
        "franka_emika_panda/assets/link0_1.obj",
        "57673d2ee2a9db2e10c8986ae57c56a7c39c012b60fd26d7d8549553b664a5ed",
        103978,
    ),
    (
        "franka_emika_panda/assets/link0_10.obj",
        "c9198b9d8534d628d097b0f2b239bb30343353182df098ce0d9db6bc13326848",
        343308,
    ),
    (
        "franka_emika_panda/assets/link0_11.obj",
        "f0adec78d6677b7799d25121981270e8bfe119b5c18d08a91ee37a54ad75f65c",
        22295,
    ),
    (
        "franka_emika_panda/assets/link0_2.obj",
        "1ca435a2a87a9608a146a47955724d43dd7ce10fc3dd3018800aa98f8ba86e05",
        590892,
    ),
    (
        "franka_emika_panda/assets/link0_3.obj",
        "3e9c6334d2b3b47944af3458ee5f20519abc20fe7ee23063bb53405ccab28046",
        47588,
    ),
    (
        "franka_emika_panda/assets/link0_4.obj",
        "215d25789704936120f119c0bc1cc7f68516dd3f8f1df61457b71acac3873f26",
        211200,
    ),
    (
        "franka_emika_panda/assets/link0_5.obj",
        "9361b104e93743045107aa5b15fb1a0835ba9c1f355aae71f56403cf6a9c33f6",
        17023,
    ),
    (
        "franka_emika_panda/assets/link0_7.obj",
        "2dc0b8ebbbc80e4c065d71e195adbc384f673438641fac82b07b801b4d2837cb",
        30699,
    ),
    (
        "franka_emika_panda/assets/link0_8.obj",
        "3b8c436999db5586b8eb47bed2b0aa7de6993868b2681b4c04246eeac42fdb7a",
        3263194,
    ),
    (
        "franka_emika_panda/assets/link0_9.obj",
        "45178f97d2b5da6bce11f610212c4ef7d08c73db33a4fc38e1b7f66b93d023f5",
        105629,
    ),
    (
        "franka_emika_panda/assets/link1.obj",
        "51091e776b16babdbe2f3849b5c041f921d1f11dd94ac19a0f1c81abf496fbae",
        3274995,
    ),
    (
        "franka_emika_panda/assets/link1.stl",
        "e41a39a94108fcf56aacff603fc91ec80541f4c1af17b51a0de5617f5566e6d2",
        15084,
    ),
    (
        "franka_emika_panda/assets/link2.obj",
        "5c9f8e715de52c3eb71b475de35993b3b96081e02abea9f28c414768877fc59b",
        3295608,
    ),
    (
        "franka_emika_panda/assets/link2.stl",
        "370f7605a0fae3529db169ded50f52f171024aa792d4d773bc84197301f6a039",
        15084,
    ),
    (
        "franka_emika_panda/assets/link3.stl",
        "0a8d638b9349c6c0eefc4e888636ac4838c4b27170f18a51699321118af709c1",
        15084,
    ),
    (
        "franka_emika_panda/assets/link3_0.obj",
        "ce2da87f51708e11b189249d41401065dabe89a06e251185a06cef4d5ff27178",
        3046804,
    ),
    (
        "franka_emika_panda/assets/link3_1.obj",
        "2880b7753a7173f38fa87ee21da4dfc7fa7be46fbe07bac5aef9e128513cb118",
        65260,
    ),
    (
        "franka_emika_panda/assets/link3_2.obj",
        "670cff81ae84693769ca4ae658c154cf909d81167e7c8d22b70ea977f2504efa",
        83591,
    ),
    (
        "franka_emika_panda/assets/link3_3.obj",
        "30ede0c812efd0e0b767d134a4eae0b84e0fa407bb6325a5a555e0569b3a3ffd",
        457684,
    ),
    (
        "franka_emika_panda/assets/link4.stl",
        "0180ebb5772ec9840cb049750cffb29a9ddc90311752a16ea34757782ef9e48d",
        15084,
    ),
    (
        "franka_emika_panda/assets/link4_0.obj",
        "443bf6d178177cf347951ecf107ccd10ca444e605e375b67304579e5f948806f",
        83486,
    ),
    (
        "franka_emika_panda/assets/link4_1.obj",
        "ca935ec7cc8b3f0e86aa079c76b0b3bf593b1e9830bab0fa7131ac51577c74f0",
        3148001,
    ),
    (
        "franka_emika_panda/assets/link4_2.obj",
        "c89caf26d379c35bb9e191ee2d91d09ab29ecc792b48d8bc10b39e5b38222175",
        455522,
    ),
    (
        "franka_emika_panda/assets/link4_3.obj",
        "1495d17f45bdb230cf58b0beef9da4e8c9cfe76a3d2d38da89017efe5e093136",
        67497,
    ),
    (
        "franka_emika_panda/assets/link5_0.obj",
        "a3238b8f2af0f2eb66d9fe63dc5d41ac52d5d38507241b5ed10e3cb019ab3e71",
        824158,
    ),
    (
        "franka_emika_panda/assets/link5_1.obj",
        "9d994bbf83ec241e301b5197a6846da13f2a4a0582edb2c0f76962c5549649d6",
        63174,
    ),
    (
        "franka_emika_panda/assets/link5_2.obj",
        "83b4ffe9f772cd7fe182e98640650eebc899aaf01842656574b0416c90dd40a4",
        3856621,
    ),
    (
        "franka_emika_panda/assets/link5_collision_0.obj",
        "3e5aa75fb13f193555c3dfb999df91d81df25006a5c3af5338297d78137c129b",
        3861,
    ),
    (
        "franka_emika_panda/assets/link5_collision_1.obj",
        "87cd8f621071df010f774cbde789282b8463840dfe704f8bdfa2e29be0142e15",
        2412,
    ),
    (
        "franka_emika_panda/assets/link5_collision_2.obj",
        "823ab6c8f7b1d79e0844fa071d82a03499ff68e34fddb1ba3451032c22736040",
        3786,
    ),
    (
        "franka_emika_panda/assets/link6.stl",
        "20b768e99a0e0440b5754dcca108016434e57937cc356acd9c352ccd3cb27f77",
        10084,
    ),
    (
        "franka_emika_panda/assets/link6_0.obj",
        "0504c70ebf1b284ee8faae7dcb896b8830bc672a6552015dd5dbd121fa031fb9",
        157563,
    ),
    (
        "franka_emika_panda/assets/link6_1.obj",
        "45f7ca77695198b6d86ec2417ad4e42e6275044f0f96266699ee8e1057757ac1",
        27106,
    ),
    (
        "franka_emika_panda/assets/link6_10.obj",
        "365f26c21d10332cab8d8704645cb46930c30b3085956f2d097498c793f24db4",
        368502,
    ),
    (
        "franka_emika_panda/assets/link6_11.obj",
        "8ae61e08b3ad064f25bbf03b22e59cc15d65883b5a7ce92711aff29df2493901",
        32653,
    ),
    (
        "franka_emika_panda/assets/link6_12.obj",
        "f28d1bfd62b0244ffd19fb35fe9edefdf583d3c9167db944e87185d919e1d021",
        3894,
    ),
    (
        "franka_emika_panda/assets/link6_13.obj",
        "0a55fa92dccb261d07200e6565fca4c850fd17f0c1ecf38821c0b97d51796c7d",
        3798,
    ),
    (
        "franka_emika_panda/assets/link6_14.obj",
        "750ae3ee4781e62e3a045c662195247b03f6642b05aa93104f08258e5f6eba26",
        444469,
    ),
    (
        "franka_emika_panda/assets/link6_15.obj",
        "5999c1c2497a004a7c4bbe034cdf13744f839d85ae7a11b1bc386c48a3e642e6",
        668552,
    ),
    (
        "franka_emika_panda/assets/link6_16.obj",
        "551d7a8b37c53e7a152d008b4e36992e5b5abe74a4be18b57ee6319c27def09f",
        3675358,
    ),
    (
        "franka_emika_panda/assets/link6_2.obj",
        "e54eee4d8232947ee8af65afd5a067e09dac2da6e7e585cb1dccb65496457da0",
        9916,
    ),
    (
        "franka_emika_panda/assets/link6_3.obj",
        "d4f6a25ef32fa1fa4734146981994ee760232cb077c80fea6487d78a077c7f27",
        11943,
    ),
    (
        "franka_emika_panda/assets/link6_4.obj",
        "be353480fe71d46f7a96f0a3cbb7e112caaf8d88368826f321a36a253c8749b2",
        13846,
    ),
    (
        "franka_emika_panda/assets/link6_5.obj",
        "7c87664623c3e1a6cb330db273227a69626dc5e53b4caff1d4ee7911eb3340c6",
        11841,
    ),
    (
        "franka_emika_panda/assets/link6_6.obj",
        "8fbc031e0d07b795ebf99115c75223141caaba317cab569159f3d477a7821681",
        12597,
    ),
    (
        "franka_emika_panda/assets/link6_7.obj",
        "7dc5ebb799fe281aab27d6e84e7a4efa0a8e42da2369ec83de3ea3f206f62d73",
        4382,
    ),
    (
        "franka_emika_panda/assets/link6_8.obj",
        "e3f77381817a341030f9dcc4c4aa7c559f80411b453fd87eb0fefecca82151bf",
        9131,
    ),
    (
        "franka_emika_panda/assets/link6_9.obj",
        "25a56ea9b02a03b6cb5a81f31c54f54d0185e1a0b3937c37e4289d896e50215a",
        17616,
    ),
    (
        "franka_emika_panda/assets/link7.stl",
        "92ac6afcf7574c034d3170d8a68e95ac9048ab9d0dd5bbd8311b86e551b9ab1c",
        10084,
    ),
    (
        "franka_emika_panda/assets/link7_0.obj",
        "04586b0bde011715cfc22072ac72fe0717eb4d4817bcf665987ce6cd88eb2b93",
        1362355,
    ),
    (
        "franka_emika_panda/assets/link7_1.obj",
        "c3f72289f394c9d6749a4a327fb11f05b203c37c95032c92f28019b40ebe7e00",
        121358,
    ),
    (
        "franka_emika_panda/assets/link7_2.obj",
        "d2faecedb304d0989040d082410378dc949dbbda23bd18c05134f08aae984db5",
        208907,
    ),
    (
        "franka_emika_panda/assets/link7_3.obj",
        "b62d87c4d4bdb7b281ed341d347c39dad0e50bf36aed8ac11d050073701d5f70",
        123684,
    ),
    (
        "franka_emika_panda/assets/link7_4.obj",
        "cb78ccb61fd4ad6c7bff4c4d2e6cb41e1964a5a3e1a28e67c0c6202371cfc018",
        85732,
    ),
    (
        "franka_emika_panda/assets/link7_5.obj",
        "099071d3090b51471ba52f4f90814fd3089f4975d196e4d488117073d0bfe491",
        226402,
    ),
    (
        "franka_emika_panda/assets/link7_6.obj",
        "99469e2d95ab74ce0292f2f16954c87a8a7dc16652e62e970fe37c94330a81bb",
        99900,
    ),
    (
        "franka_emika_panda/assets/link7_7.obj",
        "426c3db8a19373f3dc73d4bec672e948121b2914c10e2e7b9c7ea80643a19e54",
        792162,
    ),
    (
        "franka_emika_panda/panda_nohand.xml",
        "e7090a5e2384b18a223ef532d98bf914f7f9a6b74dc8bc9f74c8291201074071",
        11246,
    ),
    (
        "robotiq_2f85/2f85.xml",
        "d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de",
        9540,
    ),
    (
        "robotiq_2f85/LICENSE",
        "3d5edd18d1e22a17666e9bb84270171ed5ac90b192d24e0177fd171708ea2316",
        1297,
    ),
    (
        "robotiq_2f85/assets/base.stl",
        "1c7b9f2bd92d705fc4e897c94e905973a3c05f406845e942229433deb7041453",
        1712484,
    ),
    (
        "robotiq_2f85/assets/base_mount.stl",
        "87971dfdd6e5311b5a723e7b73c4701ebf5c2364df671500f30dfc1c53c8e61a",
        1091784,
    ),
    (
        "robotiq_2f85/assets/coupler.stl",
        "d5ee95e62f8415bf5b6e503c831a958f5fc1990bf9b2865329ec38a28932727c",
        89084,
    ),
    (
        "robotiq_2f85/assets/driver.stl",
        "bbd6e868b1778bead60d2516c7ede581d7ad4e431744b1828757d6ea5129c112",
        67084,
    ),
    (
        "robotiq_2f85/assets/follower.stl",
        "633793332c081641ce200df27a40643cc293b29956e3cb5cb29cb811c33ef1c7",
        110484,
    ),
    (
        "robotiq_2f85/assets/pad.stl",
        "4af1fa8d9bb285abecbf5dfc88e74cb30f059a7a074c2030dc35e9ee4316019d",
        15084,
    ),
    (
        "robotiq_2f85/assets/silicone_pad.stl",
        "ca20c0fa61e6d3ce7b04bed25360e048e31dfd88ace04187aa6d9e8f4ca3fd0f",
        15084,
    ),
    (
        "robotiq_2f85/assets/spring_link.stl",
        "4fb74a8a0d76c0e471cf19fd48bc676fd5b19123798e7a39eb6aa56869354283",
        84884,
    ),
)


class AssetError(RuntimeError):
    """A model file is missing, or its checksum is not correct."""


def default_cache_dir() -> Path:
    """Return the default cache directory of the simulation assets."""
    if os.environ.get("PRAXIS_SIM_CACHE"):
        return Path(os.environ["PRAXIS_SIM_CACHE"]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "praxis" / "sim"


def menagerie_dir(cache_dir: str | Path | None = None) -> Path:
    """Return the directory of the pinned Menagerie files in the cache."""
    root = (
        Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    )
    return root / f"mujoco_menagerie-{MENAGERIE_COMMIT[:12]}"


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def missing_files(root: Path) -> list[str]:
    """Return the pinned files under ``root`` that are missing or not correct.

    The check compares the size first. It computes the SHA-256 only when the
    size is correct.
    """
    bad: list[str] = []
    for rel, sha, size in FILES:
        path = root / rel
        if not path.is_file() or path.stat().st_size != size or sha256_of(path) != sha:
            bad.append(rel)
    return bad


def assets_ready(cache_dir: str | Path | None = None) -> bool:
    """Return ``True`` when all pinned files are in the cache and correct."""
    return not missing_files(menagerie_dir(cache_dir))


def _write_checked(data: bytes, rel: str, sha: str, dest: Path) -> None:
    """Check ``data`` against ``sha`` and write it to ``dest`` in one step."""
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise AssetError(f"{rel}: SHA-256 is {got}, expected {sha}.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".part-")
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp, dest)


def _fetch(rel: str, sha: str, dest: Path, source: Path | None, commit: str) -> None:
    """Get one file from a local Menagerie copy or from the pinned commit."""
    if source is not None:
        path = source / rel
        if not path.is_file():
            raise AssetError(f"{rel}: not found in {source}.")
        _write_checked(path.read_bytes(), rel, sha, dest)
        return
    url = RAW_URL.format(commit=commit, path=rel)
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AssetError(f"{rel}: download from {url} failed ({exc}).") from exc
    _write_checked(data, rel, sha, dest)


def ensure_assets(
    cache_dir: str | Path | None = None,
    source: str | Path | None = None,
    download: bool = True,
) -> Path:
    """Make sure that the pinned model files are in the cache. Return their root.

    The function gets only the files that are missing or not correct.
    ``source`` is an optional local Menagerie checkout at the pinned commit.
    With ``download=False`` and no ``source``, a missing file raises
    :class:`AssetError`. A failed download raises :class:`AssetError` with
    the instructions for a machine without a network.
    """
    root = menagerie_dir(cache_dir)
    bad = missing_files(root)
    if not bad:
        return root
    src = Path(source).expanduser() if source is not None else None
    if src is None and not download:
        raise AssetError(
            f"{len(bad)} model files are missing in {root} (first: {bad[0]}). "
            "Enable the download or give a local Menagerie copy as the source."
        )
    table = {rel: sha for rel, sha, _ in FILES}

    def job(rel: str) -> None:
        _fetch(rel, table[rel], root / rel, src, MENAGERIE_COMMIT)

    pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    try:
        for future in [pool.submit(job, rel) for rel in bad]:
            future.result()
    except AssetError as exc:
        # Stop at the first failure: do not wait for the other downloads.
        pool.shutdown(wait=False, cancel_futures=True)
        raise AssetError(
            f"Cannot get the simulation model files: {exc} "
            f"On a machine without a network, clone {MENAGERIE_REPO} at commit "
            f"{MENAGERIE_COMMIT} on a different machine, copy it here and set "
            "the asset source to that directory. Or copy the directory "
            f"{root} from a machine that has it."
        ) from exc
    pool.shutdown(wait=True)
    still = missing_files(root)
    if still:
        raise AssetError(
            f"{len(still)} model files are not correct after the download."
        )
    return root


def clear_cache(cache_dir: str | Path | None = None) -> None:
    """Remove the pinned Menagerie files from the cache."""
    root = menagerie_dir(cache_dir)
    if root.is_dir():
        shutil.rmtree(root)


def attribution() -> str:
    """Return one line of attribution text per robot model."""
    lines = [
        f"{m.name}: {MENAGERIE_REPO}/tree/{MENAGERIE_COMMIT}/{m.directory} "
        f"({m.license}, {m.copyright}; from {m.upstream})"
        for m in MODELS.values()
    ]
    return "\n".join(lines)
