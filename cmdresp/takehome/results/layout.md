# Word layouts (32-bit stream words), from serializing instances

| Message | Words | Example | Fields |
| --- | --- | --- | --- |
| CmdHdr (NTT) | 1 | 0x10123401 | `cmd_type` bits 7:0, `tx_id` bits 23:8, `nbatch` bits 31:24 |
| CmdHdr (END) | 1 | 0x00000002 | cmd_type = 2 in bits 7:0 |
| RespHdr | 1 | 0x00101234 | `tx_id` bits 15:0, `nbatch` bits 23:16 |
| Coefficient burst | 256 per polynomial | 0, 1, q-1, 0xFFFFFFFF -> 0x00000000, 0x00000001, 0x007FE000, 0xFFFFFFFF | one coefficient per word, the whole word (uint32); TLAST on word 255 |

Example values: NTT header with tx_id = 0x1234, nbatch = 16; response header for it.
