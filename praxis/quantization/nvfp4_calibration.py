"""Frozen FC1 output scales for supported NVFP4 checkpoints.

The key is the SHA256SUMS/index identity used by the packed-weight cache.
The `praxis-1-droid-nvfp4` scales use 2 * hidden amax / 2688 across 18 task scenes at seed
1000. Evaluation scenes 0 through 15 were excluded from calibration.
"""

FC1_OUTPUT_SCALES = {
    "4ae68fe4aa482abd": {
        "backbone.encoder.core.blocks.2.ffn": 4.119047619047619,
        "backbone.encoder.core.blocks.3.ffn": 0.19791666666666666,
        "backbone.encoder.core.blocks.4.ffn": 0.8392857142857143,
        "backbone.encoder.core.blocks.5.ffn": 3.4523809523809526,
        "backbone.encoder.core.blocks.6.ffn": 0.38095238095238093,
        "backbone.encoder.core.blocks.7.ffn": 0.37797619047619047,
        "backbone.encoder.core.blocks.8.ffn": 0.34375,
        "backbone.encoder.core.blocks.9.ffn": 1.3869047619047619,
        "backbone.encoder.core.blocks.10.ffn": 0.06212797619047619,
        "backbone.encoder.core.blocks.11.ffn": 1.1845238095238095,
        "backbone.encoder.core.blocks.12.ffn": 0.4732142857142857,
        "backbone.encoder.core.blocks.13.ffn": 0.12723214285714285,
        "backbone.decoder.core.blocks.1.ffn": 0.7738095238095238,
        "backbone.decoder.core.blocks.2.ffn": 0.6220238095238095,
        "backbone.decoder.core.blocks.3.ffn": 1.1547619047619047,
        "backbone.decoder.core.blocks.4.ffn": 1.1666666666666667,
        "backbone.decoder.core.blocks.5.ffn": 0.29464285714285715,
    }
}

# Bind calibration to the validated source tensors as well as the manifest.
FC1_SOURCE_DIGESTS = {
    "4ae68fe4aa482abd": {
        "backbone.decoder.core.blocks.1.ffn": (
            "c3ad200f02788ec41d7e516b6c433119260e5b5abc5c8ac7affb62b0296be13a",
            "ee93bd385d984f11adecdc3fe90693a9b08a14507b40efb8fd963e9e5efdcfdc",
        ),
        "backbone.decoder.core.blocks.2.ffn": (
            "5475efd9da4218b520b44fde528d515c412dedda5d747338e79fd3e440be26fd",
            "74f0e90a9ba59066769fd757d1b4308c688285153c01ca813908bba50cd8959f",
        ),
        "backbone.decoder.core.blocks.3.ffn": (
            "7d7e9d9275ba43fadcfd28be97767bdbce833f114e74fc3940bebb1607dc7738",
            "7bb5d1c6c84fbc1fd311066fdfbe633004f053943a0bd593a9cd115b45dd5798",
        ),
        "backbone.decoder.core.blocks.4.ffn": (
            "eb40ca93d89f6a8ca8058ac285208bf02d9c78a34f40b2b9569390a67d687192",
            "cbe895278bdd8aa835dd0713864eccb1d8eed33be74f42d11ea9bae137b7d42f",
        ),
        "backbone.decoder.core.blocks.5.ffn": (
            "5750f01aa29c9bee34007177d4d6cb3171177391c9176ed1ede508c799b51207",
            "d31f83570eea022b58df5d2644f82e390e8e1b609a8055c5904c0275f0a41e35",
        ),
        "backbone.encoder.core.blocks.10.ffn": (
            "0d1a42bdc1fd020f7d54ce1782a0c218c600c141fc15ef9c7e9addec8f657b67",
            "5c1b294d2fcd5719e6166267b67124bb4a8e94ccf82b53aa1e0c7da9967790c0",
        ),
        "backbone.encoder.core.blocks.11.ffn": (
            "67e89db2c626cf32be1b283aaaa4e8473728d238a66695f7b7c05ab1971b44ab",
            "152d08c0793ee894e827e9664100be417e5be1243b312e0dd2c12edd06c895dc",
        ),
        "backbone.encoder.core.blocks.12.ffn": (
            "37c4b3c17b48ded290d98c8f8500d7a4557e4dad424435815661b1bda33e76ed",
            "18a7edb549ae6b37f16438fe24267f3f7f75c1d644bd5e0767bb2271f552e469",
        ),
        "backbone.encoder.core.blocks.13.ffn": (
            "d20894063841ff70c8170579c4798199d338492f2c60aead68f023b5c77fa446",
            "b8a0ff4dc65d78e98746dc27343404b0e003d7181d61d433c14f66d4a77da94f",
        ),
        "backbone.encoder.core.blocks.2.ffn": (
            "ec303b0996a7b16ae6c3e6cfbc6b6d0f39aa7e3b134c8759d96e6962368c04f7",
            "378e6fa1a4bddeb7f1cd280317dc441610592b570e6d52c4a922c372f7489655",
        ),
        "backbone.encoder.core.blocks.3.ffn": (
            "1274b9fc5d1ec53bef46ba0c5ea280bd767330448467222602f73e9a95971eb5",
            "2e67bdf7868294b274338b60d2ebe7b2de310b9281e989127cdf1f0a969d3efe",
        ),
        "backbone.encoder.core.blocks.4.ffn": (
            "abdf77f312d29f04d11540ee609bb1109268ed4504048ff1569a4d4fbdd9289f",
            "b42ff382acf3fcd17657b3514810766406925b42f75afbf7d798e59a69f1f6e0",
        ),
        "backbone.encoder.core.blocks.5.ffn": (
            "4968b487c37bfc7948945e0961663b294bf4e5ad142b588575266ac09485e1c3",
            "760b29d9fc875474890ed591910e0d642de428ae4f3a3bb551cd406fce262413",
        ),
        "backbone.encoder.core.blocks.6.ffn": (
            "00a54b5cda0a51810517ff22d16668cc79b2514ca5a3d35f7c23b14394db03b2",
            "38cf4fe239bda158660c2e35e600091e5c67f7263ac14e8d0342441f7c07f4cb",
        ),
        "backbone.encoder.core.blocks.7.ffn": (
            "275266fc468dbeb652d6598582b814baaaadcee9ac8efe0cb0a8407fbb63201a",
            "e254d84441502c498f2efc7cc62731ada58adbf34ccec2befdeb1a39d0865ed8",
        ),
        "backbone.encoder.core.blocks.8.ffn": (
            "89a5addee4cdf7a0ac4cbff40b9e20025815e6b21720a292cf2d5f3ee5e54c78",
            "8eb7289573241244f489c28ab0d914466a16a2e771a9271922120e16e7624906",
        ),
        "backbone.encoder.core.blocks.9.ffn": (
            "f8dbd18028864c3c8b12ccbd0e16d977bfb0c5ea25746150ccaab196258f9e24",
            "18977925fa593045839b104c15d7a911085e6e851879feb2dc81ff4a860d2309",
        ),
    }
}
