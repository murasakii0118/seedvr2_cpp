// Text embeddings, compiled into the binary.
//
// The original data lives in seedvr2_videoupscaler/pos_emb.pt / neg_emb.pt
// (bfloat16, 58x5120 / 64x5120).  It is constant for every image, so it is
// generated into src/core/text_emb_data.h (as ``double``) and linked in
// directly - no external file is read at runtime.
#include "core/text_emb.h"
#include "core/common.h"
#include "core/text_emb_data.h"

namespace seedvr2 {

// The model consumes f32; the table itself is stored as double, so cast once.
static torch::Tensor emb_from_double(const double* p, int rows, int cols) {
    torch::Tensor t = torch::from_blob(const_cast<double*>(p), { rows, cols },
                                       torch::TensorOptions().dtype(torch::kFloat64));
    return t.to(torch::kFloat32).contiguous();
}

TextEmbeddings load_text_embeddings(const std::string& device) {
    TextEmbeddings e;
    e.pos = emb_from_double(kPosEmb, kPosEmbRows, kPosEmbCols);
    e.neg = emb_from_double(kNegEmb, kNegEmbRows, kNegEmbCols);
    if (device != "cpu") { e.pos = e.pos.to(device); e.neg = e.neg.to(device); }
    SEEDVR_LOG(LogLevel::Info, "text embeddings (hard-coded): pos " +
                                   std::to_string(e.pos.size(0)) + "x" + std::to_string(e.pos.size(1)) +
                                   ", neg " + std::to_string(e.neg.size(0)) + "x" + std::to_string(e.neg.size(1)));
    return e;
}

} // namespace seedvr2
