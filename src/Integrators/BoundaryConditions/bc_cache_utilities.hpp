#pragma once

namespace Prandtl
{

  enum class BCType : int
    {
      Invalid = -1,
      SlipWall = 0,
      SupersonicInflow = 1,
      SupersonicOutflow = 2,
      PrescribedState = 3,
      Symmetry = 4,
      Axis = 5,
      NoSlipWall = 6,
      NumBCTypes = 7
    };

  enum class BCDataKind : int
    {
      None = 0,
      ScalarConstant = 1,
      VectorConstant = 2,
      NumBCDataKinds = 3
    };

  struct BCDescriptor
  {
    int type;       // BCType
    int data_kind;  // BCDataKind
    int data_index; // offset/index into packed scalar/vector tables
    int flags;      // reserved for options
    int bdr_attr;   // optional/debug/support
    int rsrv; // alignment/expansion
  };

} // namespace Prandtl
