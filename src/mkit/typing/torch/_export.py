# This file is @generated by templates/typing/torch/_export.py.jinja.
# Do not edit.
from jaxtyping import Bool, Float, Integer, Shaped

from mkit.typing.array import (
    ArrayLike,
    B2ELike,
    B2Like,
    B2MLike,
    B2NLike,
    B2TLike,
    B2VLike,
    B3ELike,
    B3Like,
    B3MLike,
    B3NLike,
    B3TLike,
    B3VLike,
    B4ELike,
    B4Like,
    B4MLike,
    B4NLike,
    B4TLike,
    B4VLike,
    B5ELike,
    B5Like,
    B5MLike,
    B5NLike,
    B5TLike,
    B5VLike,
    B6ELike,
    B6Like,
    B6MLike,
    B6NLike,
    B6TLike,
    B6VLike,
    B22Like,
    B23Like,
    B24Like,
    B25Like,
    B26Like,
    B32Like,
    B33Like,
    B34Like,
    B35Like,
    B36Like,
    B42Like,
    B43Like,
    B44Like,
    B45Like,
    B46Like,
    B52Like,
    B53Like,
    B54Like,
    B55Like,
    B56Like,
    B62Like,
    B63Like,
    B64Like,
    B65Like,
    B66Like,
    BE2Like,
    BE3Like,
    BE4Like,
    BE5Like,
    BE6Like,
    BEELike,
    BELike,
    BEMLike,
    BENLike,
    BETLike,
    BEVLike,
    BLike,
    BM2Like,
    BM3Like,
    BM4Like,
    BM5Like,
    BM6Like,
    BMELike,
    BMLike,
    BMMLike,
    BMNLike,
    BMTLike,
    BMVLike,
    BN2Like,
    BN3Like,
    BN4Like,
    BN5Like,
    BN6Like,
    BNELike,
    BNLike,
    BNMLike,
    BNNLike,
    BNTLike,
    BNVLike,
    BT2Like,
    BT3Like,
    BT4Like,
    BT5Like,
    BT6Like,
    BTELike,
    BTLike,
    BTMLike,
    BTNLike,
    BTTLike,
    BTVLike,
    BV2Like,
    BV3Like,
    BV4Like,
    BV5Like,
    BV6Like,
    BVELike,
    BVLike,
    BVMLike,
    BVNLike,
    BVTLike,
    BVVLike,
    F2ELike,
    F2Like,
    F2MLike,
    F2NLike,
    F2TLike,
    F2VLike,
    F3ELike,
    F3Like,
    F3MLike,
    F3NLike,
    F3TLike,
    F3VLike,
    F4ELike,
    F4Like,
    F4MLike,
    F4NLike,
    F4TLike,
    F4VLike,
    F5ELike,
    F5Like,
    F5MLike,
    F5NLike,
    F5TLike,
    F5VLike,
    F6ELike,
    F6Like,
    F6MLike,
    F6NLike,
    F6TLike,
    F6VLike,
    F22Like,
    F23Like,
    F24Like,
    F25Like,
    F26Like,
    F32Like,
    F33Like,
    F34Like,
    F35Like,
    F36Like,
    F42Like,
    F43Like,
    F44Like,
    F45Like,
    F46Like,
    F52Like,
    F53Like,
    F54Like,
    F55Like,
    F56Like,
    F62Like,
    F63Like,
    F64Like,
    F65Like,
    F66Like,
    FE2Like,
    FE3Like,
    FE4Like,
    FE5Like,
    FE6Like,
    FEELike,
    FELike,
    FEMLike,
    FENLike,
    FETLike,
    FEVLike,
    FLike,
    FM2Like,
    FM3Like,
    FM4Like,
    FM5Like,
    FM6Like,
    FMELike,
    FMLike,
    FMMLike,
    FMNLike,
    FMTLike,
    FMVLike,
    FN2Like,
    FN3Like,
    FN4Like,
    FN5Like,
    FN6Like,
    FNELike,
    FNLike,
    FNMLike,
    FNNLike,
    FNTLike,
    FNVLike,
    FT2Like,
    FT3Like,
    FT4Like,
    FT5Like,
    FT6Like,
    FTELike,
    FTLike,
    FTMLike,
    FTNLike,
    FTTLike,
    FTVLike,
    FV2Like,
    FV3Like,
    FV4Like,
    FV5Like,
    FV6Like,
    FVELike,
    FVLike,
    FVMLike,
    FVNLike,
    FVTLike,
    FVVLike,
    I2ELike,
    I2Like,
    I2MLike,
    I2NLike,
    I2TLike,
    I2VLike,
    I3ELike,
    I3Like,
    I3MLike,
    I3NLike,
    I3TLike,
    I3VLike,
    I4ELike,
    I4Like,
    I4MLike,
    I4NLike,
    I4TLike,
    I4VLike,
    I5ELike,
    I5Like,
    I5MLike,
    I5NLike,
    I5TLike,
    I5VLike,
    I6ELike,
    I6Like,
    I6MLike,
    I6NLike,
    I6TLike,
    I6VLike,
    I22Like,
    I23Like,
    I24Like,
    I25Like,
    I26Like,
    I32Like,
    I33Like,
    I34Like,
    I35Like,
    I36Like,
    I42Like,
    I43Like,
    I44Like,
    I45Like,
    I46Like,
    I52Like,
    I53Like,
    I54Like,
    I55Like,
    I56Like,
    I62Like,
    I63Like,
    I64Like,
    I65Like,
    I66Like,
    IE2Like,
    IE3Like,
    IE4Like,
    IE5Like,
    IE6Like,
    IEELike,
    IELike,
    IEMLike,
    IENLike,
    IETLike,
    IEVLike,
    ILike,
    IM2Like,
    IM3Like,
    IM4Like,
    IM5Like,
    IM6Like,
    IMELike,
    IMLike,
    IMMLike,
    IMNLike,
    IMTLike,
    IMVLike,
    IN2Like,
    IN3Like,
    IN4Like,
    IN5Like,
    IN6Like,
    INELike,
    INLike,
    INMLike,
    INNLike,
    INTLike,
    INVLike,
    IT2Like,
    IT3Like,
    IT4Like,
    IT5Like,
    IT6Like,
    ITELike,
    ITLike,
    ITMLike,
    ITNLike,
    ITTLike,
    ITVLike,
    IV2Like,
    IV3Like,
    IV4Like,
    IV5Like,
    IV6Like,
    IVELike,
    IVLike,
    IVMLike,
    IVNLike,
    IVTLike,
    IVVLike,
    is_array_like,
)

__all__ = [
    "ArrayLike",
    "B2ELike",
    "B2Like",
    "B2MLike",
    "B2NLike",
    "B2TLike",
    "B2VLike",
    "B3ELike",
    "B3Like",
    "B3MLike",
    "B3NLike",
    "B3TLike",
    "B3VLike",
    "B4ELike",
    "B4Like",
    "B4MLike",
    "B4NLike",
    "B4TLike",
    "B4VLike",
    "B5ELike",
    "B5Like",
    "B5MLike",
    "B5NLike",
    "B5TLike",
    "B5VLike",
    "B6ELike",
    "B6Like",
    "B6MLike",
    "B6NLike",
    "B6TLike",
    "B6VLike",
    "B22Like",
    "B23Like",
    "B24Like",
    "B25Like",
    "B26Like",
    "B32Like",
    "B33Like",
    "B34Like",
    "B35Like",
    "B36Like",
    "B42Like",
    "B43Like",
    "B44Like",
    "B45Like",
    "B46Like",
    "B52Like",
    "B53Like",
    "B54Like",
    "B55Like",
    "B56Like",
    "B62Like",
    "B63Like",
    "B64Like",
    "B65Like",
    "B66Like",
    "BE2Like",
    "BE3Like",
    "BE4Like",
    "BE5Like",
    "BE6Like",
    "BEELike",
    "BELike",
    "BEMLike",
    "BENLike",
    "BETLike",
    "BEVLike",
    "BLike",
    "BM2Like",
    "BM3Like",
    "BM4Like",
    "BM5Like",
    "BM6Like",
    "BMELike",
    "BMLike",
    "BMMLike",
    "BMNLike",
    "BMTLike",
    "BMVLike",
    "BN2Like",
    "BN3Like",
    "BN4Like",
    "BN5Like",
    "BN6Like",
    "BNELike",
    "BNLike",
    "BNMLike",
    "BNNLike",
    "BNTLike",
    "BNVLike",
    "BT2Like",
    "BT3Like",
    "BT4Like",
    "BT5Like",
    "BT6Like",
    "BTELike",
    "BTLike",
    "BTMLike",
    "BTNLike",
    "BTTLike",
    "BTVLike",
    "BV2Like",
    "BV3Like",
    "BV4Like",
    "BV5Like",
    "BV6Like",
    "BVELike",
    "BVLike",
    "BVMLike",
    "BVNLike",
    "BVTLike",
    "BVVLike",
    "Bool",
    "F2ELike",
    "F2Like",
    "F2MLike",
    "F2NLike",
    "F2TLike",
    "F2VLike",
    "F3ELike",
    "F3Like",
    "F3MLike",
    "F3NLike",
    "F3TLike",
    "F3VLike",
    "F4ELike",
    "F4Like",
    "F4MLike",
    "F4NLike",
    "F4TLike",
    "F4VLike",
    "F5ELike",
    "F5Like",
    "F5MLike",
    "F5NLike",
    "F5TLike",
    "F5VLike",
    "F6ELike",
    "F6Like",
    "F6MLike",
    "F6NLike",
    "F6TLike",
    "F6VLike",
    "F22Like",
    "F23Like",
    "F24Like",
    "F25Like",
    "F26Like",
    "F32Like",
    "F33Like",
    "F34Like",
    "F35Like",
    "F36Like",
    "F42Like",
    "F43Like",
    "F44Like",
    "F45Like",
    "F46Like",
    "F52Like",
    "F53Like",
    "F54Like",
    "F55Like",
    "F56Like",
    "F62Like",
    "F63Like",
    "F64Like",
    "F65Like",
    "F66Like",
    "FE2Like",
    "FE3Like",
    "FE4Like",
    "FE5Like",
    "FE6Like",
    "FEELike",
    "FELike",
    "FEMLike",
    "FENLike",
    "FETLike",
    "FEVLike",
    "FLike",
    "FM2Like",
    "FM3Like",
    "FM4Like",
    "FM5Like",
    "FM6Like",
    "FMELike",
    "FMLike",
    "FMMLike",
    "FMNLike",
    "FMTLike",
    "FMVLike",
    "FN2Like",
    "FN3Like",
    "FN4Like",
    "FN5Like",
    "FN6Like",
    "FNELike",
    "FNLike",
    "FNMLike",
    "FNNLike",
    "FNTLike",
    "FNVLike",
    "FT2Like",
    "FT3Like",
    "FT4Like",
    "FT5Like",
    "FT6Like",
    "FTELike",
    "FTLike",
    "FTMLike",
    "FTNLike",
    "FTTLike",
    "FTVLike",
    "FV2Like",
    "FV3Like",
    "FV4Like",
    "FV5Like",
    "FV6Like",
    "FVELike",
    "FVLike",
    "FVMLike",
    "FVNLike",
    "FVTLike",
    "FVVLike",
    "Float",
    "I2ELike",
    "I2Like",
    "I2MLike",
    "I2NLike",
    "I2TLike",
    "I2VLike",
    "I3ELike",
    "I3Like",
    "I3MLike",
    "I3NLike",
    "I3TLike",
    "I3VLike",
    "I4ELike",
    "I4Like",
    "I4MLike",
    "I4NLike",
    "I4TLike",
    "I4VLike",
    "I5ELike",
    "I5Like",
    "I5MLike",
    "I5NLike",
    "I5TLike",
    "I5VLike",
    "I6ELike",
    "I6Like",
    "I6MLike",
    "I6NLike",
    "I6TLike",
    "I6VLike",
    "I22Like",
    "I23Like",
    "I24Like",
    "I25Like",
    "I26Like",
    "I32Like",
    "I33Like",
    "I34Like",
    "I35Like",
    "I36Like",
    "I42Like",
    "I43Like",
    "I44Like",
    "I45Like",
    "I46Like",
    "I52Like",
    "I53Like",
    "I54Like",
    "I55Like",
    "I56Like",
    "I62Like",
    "I63Like",
    "I64Like",
    "I65Like",
    "I66Like",
    "IE2Like",
    "IE3Like",
    "IE4Like",
    "IE5Like",
    "IE6Like",
    "IEELike",
    "IELike",
    "IEMLike",
    "IENLike",
    "IETLike",
    "IEVLike",
    "ILike",
    "IM2Like",
    "IM3Like",
    "IM4Like",
    "IM5Like",
    "IM6Like",
    "IMELike",
    "IMLike",
    "IMMLike",
    "IMNLike",
    "IMTLike",
    "IMVLike",
    "IN2Like",
    "IN3Like",
    "IN4Like",
    "IN5Like",
    "IN6Like",
    "INELike",
    "INLike",
    "INMLike",
    "INNLike",
    "INTLike",
    "INVLike",
    "IT2Like",
    "IT3Like",
    "IT4Like",
    "IT5Like",
    "IT6Like",
    "ITELike",
    "ITLike",
    "ITMLike",
    "ITNLike",
    "ITTLike",
    "ITVLike",
    "IV2Like",
    "IV3Like",
    "IV4Like",
    "IV5Like",
    "IV6Like",
    "IVELike",
    "IVLike",
    "IVMLike",
    "IVNLike",
    "IVTLike",
    "IVVLike",
    "Integer",
    "Shaped",
    "is_array_like",
]
