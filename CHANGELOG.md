# Changelog

## [0.1.0](https://github.com/liblaf/mesh-kit/compare/v0.0.1...v0.1.0) (2024-02-05)

### Features

- add `assert_shape` function to testing module ([7f2a6e9](https://github.com/liblaf/mesh-kit/commit/7f2a6e9d9d1e2aab3f2899ec170c44d186245272))
- add landmark registration functionality ([6230619](https://github.com/liblaf/mesh-kit/commit/62306198b7c9024016fb33de63ab0b32d91945eb))
- add registration functionality to sculptor ([8b215ef](https://github.com/liblaf/mesh-kit/commit/8b215ef600d8e7a23c682ffcdb8819cfd38bb8d6))
- add support for recording intermediate images during CT to mesh conversion ([dba253a](https://github.com/liblaf/mesh-kit/commit/dba253a0f8ec921af1e161feb212d2760972ceda))
- optimize correspondence calculation in nricp registration ([e65abd2](https://github.com/liblaf/mesh-kit/commit/e65abd242f04a04061ec6ccca5ec9ae80a445650))
- refactor CT_to_mesh.py, align.py, and annotate_landmarks.py ([d628199](https://github.com/liblaf/mesh-kit/commit/d62819939aee832740f10c16eec3c7c86808a186))
- refactor registration process and improve performance ([5528cfc](https://github.com/liblaf/mesh-kit/commit/5528cfc5d62c4dbbad364d63f96e894c2400ceb8))
- reorganize project structure and update dependencies ([5e68866](https://github.com/liblaf/mesh-kit/commit/5e68866d4c2863c308816aa3b2f642b5d12179c2))
- **sculptor:** add ply2ascii command ([ba951ea](https://github.com/liblaf/mesh-kit/commit/ba951ea75582a07d6f28e4225649fe65a0ad4291))
- **sculptor:** add task to preprocess template models ([5a39ad6](https://github.com/liblaf/mesh-kit/commit/5a39ad6158f7b910e27a60c33b629bc746a644bb))
- **sculptor:** update registration parameters ([e04f8b2](https://github.com/liblaf/mesh-kit/commit/e04f8b251ea122b6e505793bba7db242a2e0ad06))
- update correspondence scale and weight normal in registration ([b57b109](https://github.com/liblaf/mesh-kit/commit/b57b1096d0a96bdab68ccfa6ae4a9d2e644ad491))

### Bug Fixes

- fix connectivity parameter in CT_to_mesh.py ([eb52b8f](https://github.com/liblaf/mesh-kit/commit/eb52b8fe095b3ff2ea006876e7222ab078d9f6b4))
- fix issue with slider not updating correctly in view_records.py ([b5d4efa](https://github.com/liblaf/mesh-kit/commit/b5d4efaca21a2b849a255a6cb499f6960df39c85))
- **register:** update weight_smooth and weight_landmark values ([d297090](https://github.com/liblaf/mesh-kit/commit/d297090e83a6f208a8044638aacc1b7847e2fe8b))

## 0.0.1 (2023-10-28)

### Features

- add sculptor example code and scripts ([a51182b](https://github.com/liblaf/mesh-kit/commit/a51182b782ef65a928b1a382a841f1ec7e2884fe))
